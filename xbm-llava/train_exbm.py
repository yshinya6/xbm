# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
import datetime
import json
import logging
import math
import os
import pdb
import random
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import clip
import numpy as np
import tokenizers
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import transformers
from packaging import version
from torchvision.transforms.functional import resize
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

import utils
from data import create_dataset, create_loader, create_sampler
from llava import conversation as conversation_lib
from llava.model.llava_exbm import LlavaExbm
from utils import cosine_lr_schedule, warmup_lr_schedule

IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse("0.14")

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default="linear")
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default="flat")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    num_classes: int = field(default=1000)
    max_decode_length: int = field(default=50)
    min_decode_length: int = field(default=20)
    fixed_caps: bool = field(default=False)


@dataclass
class DataArguments:
    dataset: str = field(default="imagenet")
    image_size: int = field(default=336)
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = "square"


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    double_quant: bool = field(
        default=True, metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4", metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    checkpoint: Optional[str] = field(default=None)
    output_dir: Optional[str] = field(default="result/temp")
    evaluate: Optional[bool] = field(default=False)
    max_epoch: Optional[int] = field(default=100)
    init_lr: Optional[float] = field(default=3e-5)
    min_lr: Optional[float] = field(default=0.0)
    lr_decay_rate: Optional[float] = field(default=0.9)
    warmup_step: Optional[int] = field(default=50)
    warmup_lr: Optional[float] = field(default=1e-6)
    cuda_device: Optional[str] = field(default="cuda")
    lambda_: Optional[float] = field(default=1.0)
    temperature: Optional[float] = field(default=10.0)
    temperature_annealing: Optional[str] = field(default="const")
    distributed: Optional[bool] = field(default=False)
    dist_url: Optional[str] = field(default="env://")
    world_size: Optional[int] = field(default=1)


class TemperatureAnnealer:
    def __init__(self, temp_init, temp_min=0.5, anneal_rate=1e-6, mode="const"):
        self.current_temp = temp_init
        self.temp_min = temp_min
        self.anneal_rate = anneal_rate
        self.global_step = 0
        self.mode = mode

    def get_annealed_temp(self):
        if self.mode == "const":
            return self.current_temp
        elif self.mode == "exp":
            self.current_temp = max(self.current_temp * math.exp(-self.anneal_rate * self.global_step), self.temp_min)
        else:
            raise NotImplementedError()
        self.global_step += 1
        return self.current_temp


def train(model, data_loader, optimizer, epoch, device, t_annealer, training_args):
    # train
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.8f}"))
    metric_logger.add_meter("temperature", utils.SmoothedValue(window_size=1, fmt="{value:.8f}"))
    metric_logger.add_meter("loss_cls", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    metric_logger.add_meter("loss_lm", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    metric_logger.add_meter("train_accuracy", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))

    header = "Train Epoch: [{}]".format(epoch)
    print_freq = 10
    dtype = torch.bfloat16 if training_args.bf16 else torch.float16
    data_loader.sampler.set_epoch(epoch)
    for i, (image, label) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        if epoch == 0:
            warmup_lr_schedule(optimizer, i, training_args.warmup_step, training_args.warmup_lr, training_args.init_lr)

        optimizer.zero_grad()
        image = image.to(dtype=dtype, device=device, non_blocking=True)
        label = label.to(device=device, non_blocking=True)
        temperature = t_annealer.get_annealed_temp()
        loss_cls, loss_lm, accuracy = model(image, label, temperature=temperature)
        loss = loss_cls + training_args.lambda_ * loss_lm if training_args.lambda_ > 0 else loss_cls
        loss.backward()
        optimizer.step()
        metric_logger.update(loss_cls=loss_cls.item())
        metric_logger.update(loss_lm=loss_lm.item())
        metric_logger.update(train_accuracy=accuracy.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(temperature=temperature)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


def tokenize_clip(text_batch, prefix="A photo depicts"):
    tokeneized_text_batch = []
    for text in text_batch:
        tokenized = clip.tokenize(prefix + text, truncate=True)
        tokeneized_text_batch.append(tokenized)
    return torch.cat(tokeneized_text_batch, dim=0)


@torch.no_grad()
def get_clip_score(model, images, captions, device, w=2.5):
    if captions is None:
        return 0.0
    image_feats = model.encode_image(resize(images, (224, 224))).cpu().numpy()
    tokenized_captions = tokenize_clip(captions).to(device)
    caps_feats = model.encode_text(tokenized_captions).cpu().numpy()
    image_feats = image_feats / np.sqrt(np.sum(image_feats**2, axis=1, keepdims=True))
    caps_feats = caps_feats / np.sqrt(np.sum(caps_feats**2, axis=1, keepdims=True))

    per = w * np.clip(np.sum(image_feats * caps_feats, axis=1), 0, None)
    return np.mean(per)


@torch.no_grad()
def get_perplexity(model, tokenizer, captions, device):
    if captions is None:
        return 0.0
    input_tokens = tokenizer(captions, return_tensors="pt", padding=True)
    outputs = model(
        input_tokens.input_ids.to(device),
        attention_mask=input_tokens.attention_mask.to(device),
        labels=input_tokens.input_ids.to(device),
    )
    return torch.exp(outputs.loss).item()


@torch.no_grad()
def evaluate(model, data_loader, device, clip_model, gpt_model, gpt_tokenizer, training_args):
    # evaluate
    dtype = torch.bfloat16 if training_args.bf16 else torch.float16
    model.eval()
    clip_model = clip_model.to(device)
    gpt_model = gpt_model.to(device)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Evaluation:"
    print_freq = 10
    losses, accs, clip_scores, perplexities = [], [], [], []
    for image, label in metric_logger.log_every(data_loader, print_freq, header):

        image = image.to(dtype=dtype, device=device)
        label = label.to(device)

        loss, acc, cap = model.infer(image, class_label=label, ret_caption=True)
        clip_score = get_clip_score(clip_model, image, cap, device)
        perplexity = get_perplexity(gpt_model, gpt_tokenizer, cap, device)

        losses.append(loss.item())
        accs.append(acc.item())
        clip_scores.append(clip_score)
        perplexities.append(perplexity)
    return statistics.mean(losses), statistics.mean(accs), statistics.mean(clip_scores), statistics.mean(perplexities)


def main(model_args, data_args, training_args):
    if training_args.distributed:
        utils.init_distributed_mode(training_args)

    device = torch.device(training_args.device)

    # fix the seed for reproducibility
    seed = 42 + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    #### Model ####
    print("Creating model")
    model = LlavaExbm(model_args=model_args, training_args=training_args, data_args=data_args)
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        params=model.parameters(), lr=training_args.init_lr, weight_decay=training_args.weight_decay
    )

    #### Dataset ####
    print("Creating dataset")
    data_args.image_processor = model.text_decoder.get_vision_tower().image_processor
    data_args.multimodal = True
    train_dataset, val_dataset, test_dataset = create_dataset(data_args, min_scale=0.2)
    print("number of training samples: %d" % len(train_dataset))

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    samplers = create_sampler([train_dataset, val_dataset, test_dataset], [True, False, False], num_tasks, global_rank)
    train_loader, val_loader, test_loader = create_loader(
        [train_dataset, val_dataset, test_dataset],
        samplers,
        batch_size=[
            training_args.per_device_train_batch_size,
            training_args.per_device_eval_batch_size,
            training_args.per_device_eval_batch_size,
        ],
        num_workers=[4, 4, 4],
        is_trains=[True, False, False],
        collate_fns=[None, None, None],
    )

    # For evaluation
    clip_model, _ = clip.load("ViT-B/32")
    clip_model.eval()
    gpt_model = GPT2LMHeadModel.from_pretrained("gpt2")
    gpt_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    gpt_tokenizer.pad_token = gpt_tokenizer.eos_token

    start_epoch = 0
    if training_args.checkpoint:
        checkpoint = torch.load(training_args.checkpoint, map_location="cpu")
        state_dict = checkpoint["model"]
        model.load_state_dict(state_dict)

        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        print("resume checkpoint from %s" % training_args.checkpoint)

    if training_args.evaluate:
        start_epoch = 0
    best = 10000000
    best_epoch = 0

    model_without_ddp = model
    if training_args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[training_args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module

    t_annealer = TemperatureAnnealer(temp_init=training_args.temperature, mode=training_args.temperature_annealing)

    print("Start training")
    start_time = time.time()
    for epoch in range(start_epoch, training_args.max_epoch):
        if not training_args.evaluate:
            if training_args.distributed:
                train_loader.sampler.set_epoch(epoch)

            cosine_lr_schedule(optimizer, epoch, training_args.max_epoch, training_args.init_lr, training_args.min_lr)

            train_loss = train(model, train_loader, optimizer, epoch, device, t_annealer, training_args)

        val_loss, val_acc, val_clip_score, val_perplexity = evaluate(
            model_without_ddp, val_loader, device, clip_model, gpt_model, gpt_tokenizer, training_args
        )
        print(
            f"val_loss: {val_loss}, val_acc: {val_acc}, val_clip_score: {val_clip_score}, val_perplexity: {val_perplexity}"
        )

        test_loss, test_acc, test_clip_score, test_perplexity = evaluate(
            model_without_ddp, test_loader, device, clip_model, gpt_model, gpt_tokenizer, training_args
        )
        print(
            f"test_loss: {test_loss}, test_acc: {test_acc}, test_clip_score: {test_clip_score}, test_perplexity: {test_perplexity}"
        )

        if utils.is_main_process():
            if training_args.evaluate:
                log_stats = {
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_clip_score": float(val_clip_score),
                    "val_perplexity": float(val_perplexity),
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                    "test_clip_score": float(test_clip_score),
                    "test_perplexity": float(test_perplexity),
                }
                with open(os.path.join(training_args.output_dir, "evaluate.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")
            else:
                save_obj = {
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                }

                test_log = {}
                if val_loss < best:
                    best = val_loss
                    best_epoch = epoch
                    torch.save(save_obj, os.path.join(training_args.output_dir, "checkpoint_best.pth"))
                    test_log = {
                        "test_loss": test_loss,
                        "test_acc": test_acc,
                        "test_clip_score": float(test_clip_score),
                        "test_perplexity": float(test_perplexity),
                    }

                # if epoch % 20 == 0:
                #     torch.save(save_obj, os.path.join(training_args.output_dir, f"checkpoint_{epoch}epoch.pth"))

                log_stats = {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_clip_score": float(val_clip_score),
                    "val_perplexity": float(val_perplexity),
                    "epoch": epoch,
                    "best_epoch": best_epoch,
                    **test_log,
                }

                with open(os.path.join(training_args.output_dir, "log.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")

        if training_args.evaluate:
            break
        if training_args.distributed:
            dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    main(model_args, data_args, training_args)
