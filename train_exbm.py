import argparse
import datetime
import json
import math
import os
import pdb
import random
import statistics
import time
from pathlib import Path

import clip
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from ruamel import yaml
from torchvision.transforms.functional import resize
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

import utils
from data import create_dataset, create_loader, create_sampler
from data.utils import coco_caption_eval, save_result
from models.blip import load_checkpoint
from models.blip_exbm import blip_exbm
from utils import cosine_lr_schedule, warmup_lr_schedule

os.environ["TOKENIZERS_PARALLELISM"] = "false"


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


def train(model, data_loader, optimizer, epoch, device, config):
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
    data_loader.sampler.set_epoch(epoch)
    for i, (image, label) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        if epoch == 0:
            warmup_lr_schedule(optimizer, i, config["warmup_steps"], config["warmup_lr"], config["init_lr"])

        optimizer.zero_grad()
        image = image.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        # pdb.set_trace()  # model.module.text_decoder.bert.encoder.layer[0].intermediate.dense.weight
        temperature = config["temperature"].get_annealed_temp()
        loss_cls, loss_lm, accuracy = model(image, label, temperature=temperature)
        loss = loss_cls + config["lambda"] * loss_lm if config["lambda"] > 0 else loss_cls
        loss.backward()
        # pdb.set_trace()
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
def evaluate(model, data_loader, device, clip_model, gpt_model, gpt_tokenizer):
    # evaluate
    model.eval()
    clip_model = clip_model.to(device)
    gpt_model = gpt_model.to(device)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Evaluation:"
    print_freq = 10
    losses, accs, clip_scores, perplexities = [], [], [], []
    for image, label in metric_logger.log_every(data_loader, print_freq, header):

        image = image.to(device)
        label = label.to(device)

        loss, acc, cap = model.infer(image, class_label=label, ret_caption=True)
        clip_score = get_clip_score(clip_model, image, cap, device)
        perplexity = get_perplexity(gpt_model, gpt_tokenizer, cap, device)

        losses.append(loss.item())
        accs.append(acc.item())
        clip_scores.append(clip_score)
        perplexities.append(perplexity)
    return statistics.mean(losses), statistics.mean(accs), statistics.mean(clip_scores), statistics.mean(perplexities)


def main(args, config):
    if args.distributed:
        utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset ####
    print("Creating dataset")
    train_dataset, val_dataset, test_dataset = create_dataset(config["dataset"], config, min_scale=0.2)
    print("number of training samples: %d" % len(train_dataset))

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    samplers = create_sampler([train_dataset, val_dataset, test_dataset], [True, False, False], num_tasks, global_rank)
    train_loader, val_loader, test_loader = create_loader(
        [train_dataset, val_dataset, test_dataset],
        samplers,
        batch_size=[config["batch_size"]] * 3,
        num_workers=[4, 4, 4],
        is_trains=[True, False, False],
        collate_fns=[None, None, None],
    )

    #### Model ####
    print("Creating model")
    if "vit_only" in config:
        use_vit_only = config["vit_only"]
    else:
        use_vit_only = False
    model = blip_exbm(
        num_classes=config["num_classes"],
        image_size=config["image_size"],
        vit=config["vit"],
        vit_grad_ckpt=config["vit_grad_ckpt"],
        vit_ckpt_layer=config["vit_ckpt_layer"],
        sample=config["sample"],
        max_decode_length=config["max_decode_length"],
        min_decode_length=config["min_decode_length"],
        use_cross_attn=config["use_cross_attn"],
        fixed_caps=config["fixed_caps"],
        fixed_vit=config["fixed_vit"],
        tied_encoder=config["tied_encoder"],
        temperature=config["temperature"],
        use_vit_only=use_vit_only,
    )
    model, _ = load_checkpoint(model, config["pretrained"], args.without_encoder)
    model.copy_params()
    model = model.to(device)

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config["init_lr"], weight_decay=config["weight_decay"])

    # For evaluation
    clip_model, _ = clip.load("ViT-B/32")
    clip_model.eval()
    gpt_model = GPT2LMHeadModel.from_pretrained("gpt2")
    gpt_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    gpt_tokenizer.pad_token = gpt_tokenizer.eos_token

    start_epoch = 0
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        state_dict = checkpoint["model"]
        model.load_state_dict(state_dict)

        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        print("resume checkpoint from %s" % args.checkpoint)

    if args.evaluate:
        start_epoch = 0
    best = 10000000
    best_epoch = 0

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    if "lambda" not in config:
        config["lambda"] = 1.0

    if "temperature_annealing" not in config:
        config["temperature_annealing"] = "const"
        print("Constant temperature enabled.")

    t_annealer = TemperatureAnnealer(temp_init=config["temperature"], mode=config["temperature_annealing"])

    print("Start training")
    start_time = time.time()
    for epoch in range(start_epoch, config["max_epoch"]):
        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)

            cosine_lr_schedule(optimizer, epoch, config["max_epoch"], config["init_lr"], config["min_lr"])

            config["temperature"] = t_annealer
            train_loss = train(model, train_loader, optimizer, epoch, device, config)

        val_loss, val_acc, val_clip_score, val_perplexity = evaluate(
            model_without_ddp, val_loader, device, clip_model, gpt_model, gpt_tokenizer
        )
        print(
            f"val_loss: {val_loss}, val_acc: {val_acc}, val_clip_score: {val_clip_score}, val_perplexity: {val_perplexity}"
        )

        if val_loss < best:
            test_loss, test_acc, test_clip_score, test_perplexity = evaluate(
                model_without_ddp, test_loader, device, clip_model, gpt_model, gpt_tokenizer
            )
            print(
                f"test_loss: {test_loss}, test_acc: {test_acc}, test_clip_score: {test_clip_score}, test_perplexity: {test_perplexity}"
            )

        if utils.is_main_process():
            if args.evaluate:
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
                with open(os.path.join(args.output_dir, "evaluate.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")
            else:
                save_obj = {
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": config,
                    "epoch": epoch,
                }

                test_log = {}
                if val_loss < best:
                    best = val_loss
                    best_epoch = epoch
                    torch.save(save_obj, os.path.join(args.output_dir, "checkpoint_best.pth"))
                    test_log = {
                        "test_loss": test_loss,
                        "test_acc": test_acc,
                        "test_clip_score": float(test_clip_score),
                        "test_perplexity": float(test_perplexity),
                    }

                if epoch % 20 == 0:
                    torch.save(save_obj, os.path.join(args.output_dir, f"checkpoint_{epoch}epoch.pth"))

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

                with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")

        if args.evaluate:
            break
        if args.distributed:
            dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./configs/car.yaml")
    parser.add_argument("--output_dir", default="output/ImageClassification")
    parser.add_argument("--vit_only", action="store_true")
    parser.add_argument("--without_encoder", action="store_true")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")
    parser.add_argument("--distributed", action="store_true")
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, "config.yaml"), "w"))

    main(args, config)
