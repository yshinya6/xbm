"""
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
"""

import pdb

import transformers
from llava.model.med import BertConfig, BertLMHeadModel, BertModel

transformers.logging.set_verbosity_error()


import torch
import torch.nn.functional as F
from torch import nn
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from llava.model.language_model.llava_mpt import LlavaMptForCausalLM
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IGNORE_INDEX,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import tokenizer_image_token


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ["mm_projector", "vision_tower", "vision_resampler"]
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def create_decoder(model_args, training_args, data_args):
    compute_dtype = torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)

    if "mpt" in model_args.model_name_or_path:
        config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
        config.attn_config["attn_impl"] = "flash_attention_2"
        model = LlavaMptForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir,
        )
    else:
        model = LlavaLlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation="flash_attention_2",
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if "mpt" in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )
    tokenizer.pad_token = tokenizer.unk_token

    model.get_model().initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp)

    vision_tower = model.get_vision_tower()
    vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

    data_args.image_processor = vision_tower.image_processor
    data_args.is_multimodal = True

    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length

    model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
    if model_args.tune_mm_mlp_adapter:
        model.requires_grad_(False)
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True

    model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
    if training_args.freeze_mm_mlp_adapter:
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = False

    if training_args.bits in [4, 8]:
        model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

    model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_projector_lr = training_args.mm_projector_lr
    training_args.use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
    model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)
    return model, tokenizer


class LlavaExbm(nn.Module):

    def __init__(
        self,
        model_args,
        training_args,
        data_args,
        med_config="configs/bert_config.json",
        embed_dim=256,
        clip_dim=4096,
        prompt="Generate sentence describing details of visual objects in this image: ",
        num_beams=3,
        top_p=0.9,
        repetition_penalty=1.0,
        use_cross_attn=True,
        momentum=1.0,
        temperature=1,
    ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init__()
        self.num_classes = model_args.num_classes
        self.model_args = model_args

        # create the decoder
        self.text_decoder, self.tokenizer = create_decoder(model_args, training_args, data_args)
        self.text_decoder.config.use_cache = False
        if training_args.lora_enable:
            from peft import LoraConfig, get_peft_model

            lora_config = LoraConfig(
                r=training_args.lora_r,
                lora_alpha=training_args.lora_alpha,
                target_modules=find_all_linear_names(self.text_decoder),
                lora_dropout=training_args.lora_dropout,
                bias=training_args.lora_bias,
                task_type="CAUSAL_LM",
            )
            if training_args.bits == 16:
                if training_args.bf16:
                    self.text_decoder.to(torch.bfloat16)
                if training_args.fp16:
                    self.text_decoder.to(torch.float16)
            print("Adding LoRA adapters...")
            self.text_decoder = get_peft_model(self.text_decoder, lora_config)

        encoder_config = BertConfig.from_json_file(med_config)
        encoder_config.encoder_width = clip_dim
        self.text_encoder = BertModel.from_pretrained(
            "bert-base-uncased", config=encoder_config, add_pooling_layer=False
        )
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))

        text_width = self.text_encoder.config.hidden_size
        self.text_proj = nn.Linear(text_width, embed_dim)
        self.classifier = nn.Linear(embed_dim, self.num_classes)
        self.classifier.weight.data.normal_(mean=0.0, std=0.01)
        self.classifier.bias.data.zero_()

        # create momentum vision encoder / text decoder
        self.text_decoder_m, _ = create_decoder(model_args, training_args, data_args)
        self.model_pairs = [
            [self.text_decoder, self.text_decoder_m],
        ]
        # self.copy_params()

        if model_args.mm_use_im_start_end:
            self.prompt_base_text = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + prompt
        else:
            self.prompt_base_text = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        self.momentum = momentum
        self.num_beams = num_beams
        self.max_decode_length = model_args.max_decode_length
        self.min_decode_length = model_args.min_decode_length
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.use_cross_attn = use_cross_attn
        self.caption_context = torch.no_grad if model_args.fixed_caps else torch.enable_grad
        self.temperature = temperature

    def preprocess(self, answers):
        prompts = ["<image>\n" + prompt for prompt in answers]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [
                tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                for prompt in prompts
            ],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        targets = input_ids.clone()
        instruction_len = len(tokenizer_image_token("<image>\n", self.tokenizer))
        for target in targets:
            target[:instruction_len] = IGNORE_INDEX
        return input_ids, targets

    def tokenize(self, batch_size):
        conv = conv_templates["llava_v1"].copy()
        conv.append_message(conv.roles[0], self.prompt_base_text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        return input_ids.unsqueeze(0).repeat(batch_size, 1)

    @torch.no_grad()
    def infer(self, image, class_label=None, ret_caption=False, ret_attention=False):
        # Generate Caption
        input_ids = self.tokenize(batch_size=image.size(0)).to(image.device)
        # beam search
        decoder_output, image_embeds = self.text_decoder.generate(
            inputs=input_ids,
            images=image,
            image_sizes=[image.size],
            max_length=self.max_decode_length,
            min_length=self.min_decode_length,
            num_beams=self.num_beams,
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            repetition_penalty=self.repetition_penalty,
            output_inputs_embeds=True,
        )
        captions = []
        for output in decoder_output:
            caption = self.tokenizer.decode(output, skip_special_tokens=True)
            captions.append(caption)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        # Encode Caption (and Image Embedding)
        caption_text = self.tokenizer(
            captions,
            padding="max_length",
            truncation=True,
            max_length=self.max_decode_length,
            return_tensors="pt",
        ).to(image.device)
        model_kwargs = {"encoder_hidden_states": image_embeds.float(), "encoder_attention_mask": image_atts}
        cross_attn_kwargs = {"mode": "text"} if not self.use_cross_attn else model_kwargs
        caption_text_output = self.text_encoder(
            caption_text.input_ids,
            attention_mask=caption_text.attention_mask,
            return_dict=True,
            output_attentions=True,
            **cross_attn_kwargs,
        )
        text_feat = F.normalize(self.text_proj(caption_text_output.last_hidden_state[:, 0, :]), dim=-1)

        # Predict Final Labels
        logit = self.classifier(text_feat)
        if class_label is not None:
            loss_cls = F.cross_entropy(logit, class_label)
            acc = accuracy(logit.detach(), class_label)[0]
        else:
            loss_cls, acc = None, None

        if ret_attention:
            return logit, captions, caption_text_output
        if ret_caption:
            return loss_cls, acc, captions
        else:
            return loss_cls, acc

    def forward(self, image, class_label, ret_caption=False, temperature=None):
        # Get Pseudo Ground Truth Captions from Momentum Decoder
        with torch.no_grad():
            self._momentum_update()
            input_ids_m = self.tokenize(batch_size=image.size(0)).to(image.device)
            # beam search
            decoder_output_m = self.text_decoder_m.generate(
                inputs=input_ids_m,
                images=image,
                image_sizes=[image.size],
                max_length=self.max_decode_length,
                min_length=self.min_decode_length,
                num_beams=self.num_beams,
                eos_token_id=self.tokenizer.sep_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                repetition_penalty=self.repetition_penalty,
            )
            captions_m = []
            for output in decoder_output_m:
                caption_m = self.tokenizer.decode(output, skip_special_tokens=True)
                captions_m.append(caption_m)
        # Decode Caption from Image Embeddings
        with self.caption_context():
            decoder_input_ids, decoder_targets = self.preprocess(captions_m)
            decoder_targets = decoder_targets.masked_fill(decoder_targets == self.tokenizer.pad_token_id, -100)
            decoder_outputs, image_embeds, embeded_targets = self.text_decoder(
                input_ids=decoder_input_ids.to(image.device),
                images=image,
                labels=decoder_targets.to(image.device),
                return_dict=True,
                output_inputs_embeds=True,
            )
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
            tau = self.temperature if temperature is None else temperature
            token_mask = nn.functional.gumbel_softmax(decoder_outputs.logits, tau=tau, hard=True, dim=-1)
            indices = torch.tensor(range(0, token_mask.size(-1))).to(token_mask.device)
            decoded_tokens = (token_mask * indices).sum(-1)
            captions = []
            reduced_decoded_tokens = []
            for output, target in zip(decoded_tokens, embeded_targets):
                reduced_output = output[target != IGNORE_INDEX]
                caption = self.tokenizer.decode(reduced_output.detach(), skip_special_tokens=True)
                captions.append(caption)
                reduced_decoded_tokens.append(reduced_output)
            loss_lm = decoder_outputs.loss
            reduced_decoded_tokens = torch.nn.utils.rnn.pad_sequence(
                reduced_decoded_tokens, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
        # Encode Caption (and Image Embedding)
        caption_text = self.tokenizer(
            captions,
            padding="max_length",
            truncation=True,
            max_length=reduced_decoded_tokens.size(1),
            return_tensors="pt",
        ).to(image.device)
        model_kwargs = {"encoder_hidden_states": image_embeds.float(), "encoder_attention_mask": image_atts}
        cross_attn_kwargs = {"mode": "text"} if not self.use_cross_attn else model_kwargs
        caption_text_output = self.text_encoder(
            reduced_decoded_tokens,
            attention_mask=caption_text.attention_mask,
            return_dict=True,
            output_attentions=True,
            **cross_attn_kwargs,
        )
        text_feat = F.normalize(self.text_proj(caption_text_output.last_hidden_state[:, 0, :]), dim=-1)

        # 5. Predict Final Labels
        logit = self.classifier(text_feat)
        loss_cls = F.cross_entropy(logit, class_label)
        acc = accuracy(logit.detach(), class_label)[0]

        if ret_caption:
            return loss_cls, loss_lm, acc, captions
        else:
            return loss_cls, loss_lm, acc

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        if self.momentum == 1.0:
            return
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1.0 - self.momentum)


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
