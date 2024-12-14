import pdb

import transformers
from transformers import BertTokenizer

from models.med import BertConfig, BertLMHeadModel, BertModel

transformers.logging.set_verbosity_error()

import torch
import torch.nn.functional as F
from torch import nn

from models.blip import create_vit, init_tokenizer, load_checkpoint


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


class BLIP_ExBM(nn.Module):

    def __init__(
        self,
        num_classes,
        med_config="configs/bert_config.json",
        image_size=384,
        vit="base",
        vit_grad_ckpt=False,
        vit_ckpt_layer=0,
        embed_dim=256,
        prompt="a picture of ",
        sample=False,
        num_beams=3,
        max_decode_length=30,
        min_decode_length=10,
        top_p=0.9,
        repetition_penalty=1.0,
        use_cross_attn=False,
        fixed_caps=False,
        fixed_vit=False,
        tied_encoder=False,
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
        self.num_classes = num_classes
        self.visual_encoder, vision_width = create_vit(vit, image_size, vit_grad_ckpt, vit_ckpt_layer, 0)
        self.vision_width = vision_width
        self.tokenizer = init_tokenizer()
        encoder_config = BertConfig.from_json_file(med_config)
        encoder_config.encoder_width = vision_width
        self.text_encoder = BertModel.from_pretrained(
            "bert-base-uncased", config=encoder_config, add_pooling_layer=False
        )
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))

        text_width = self.text_encoder.config.hidden_size

        self.text_proj = nn.Linear(text_width, embed_dim)
        self.classifier = nn.Linear(embed_dim, self.num_classes)
        self.classifier.weight.data.normal_(mean=0.0, std=0.01)
        self.classifier.bias.data.zero_()

        # create the decoder
        decoder_config = BertConfig.from_json_file(med_config)
        decoder_config.encoder_width = vision_width
        self.text_decoder = BertLMHeadModel.from_pretrained("bert-base-uncased", config=decoder_config)
        self.text_decoder.resize_token_embeddings(len(self.tokenizer))
        if tied_encoder:
            tie_encoder_decoder_weights(self.text_encoder, self.text_decoder.bert, "", "/attention")

        # create momentum vision encoder / text decoder
        self.visual_encoder_m, vision_width = create_vit(vit, image_size, vit_grad_ckpt, vit_ckpt_layer, 0)
        self.text_decoder_m = BertLMHeadModel(config=decoder_config)
        self.text_decoder_m.resize_token_embeddings(len(self.tokenizer))
        self.model_pairs = [
            [self.visual_encoder, self.visual_encoder_m],
            [self.text_decoder, self.text_decoder_m],
        ]
        self.copy_params()

        self.momentum = momentum
        self.prompt = prompt
        self.sample = sample
        self.num_beams = num_beams
        self.max_decode_length = max_decode_length
        self.min_decode_length = min_decode_length
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.use_cross_attn = use_cross_attn
        self.caption_context = torch.no_grad if fixed_caps else torch.enable_grad
        self.vision_context = torch.no_grad if fixed_vit else torch.enable_grad
        self.temperature = temperature

    @torch.no_grad()
    def infer(self, image, class_label=None, ret_caption=False, ret_attention=False):
        # 1. Generate Image Embeddings
        image_embeds = self.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        # 2. Generate Caption
        with torch.no_grad():
            image_embeds_beam = image_embeds.detach().repeat_interleave(self.num_beams, dim=0)
            image_atts_beam = torch.ones(image_embeds_beam.size()[:-1], dtype=torch.long).to(image.device)
            decoder_kwargs_beam = {
                "encoder_hidden_states": image_embeds_beam,
                "encoder_attention_mask": image_atts_beam,
            }
            prompt = [self.prompt] * image.size(0)
            input_ids_m = self.tokenizer(prompt, return_tensors="pt").input_ids.to(image.device)
            input_ids_m[:, 0] = self.tokenizer.bos_token_id
            input_ids_m = input_ids_m[:, :-1]
            # beam search
            decoder_output = self.text_decoder.generate(
                input_ids=input_ids_m,
                max_length=self.max_decode_length,
                min_length=self.min_decode_length,
                num_beams=self.num_beams,
                eos_token_id=self.tokenizer.sep_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                repetition_penalty=self.repetition_penalty,
                **decoder_kwargs_beam,
            )
            captions = []
            for output in decoder_output:
                caption = self.tokenizer.decode(output, skip_special_tokens=True)
                captions.append(caption[len(self.prompt) :])

        # 3. Encode Caption (and Image Embedding)
        caption_text = self.tokenizer(
            captions,
            padding="max_length",
            truncation=True,
            max_length=self.max_decode_length,
            return_tensors="pt",
        ).to(image.device)
        model_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask": image_atts}
        cross_attn_kwargs = {"mode": "text"} if not self.use_cross_attn else model_kwargs
        caption_text_output = self.text_encoder(
            caption_text.input_ids,
            attention_mask=caption_text.attention_mask,
            return_dict=True,
            output_attentions=True,
            **cross_attn_kwargs,
        )
        text_feat = F.normalize(self.text_proj(caption_text_output.last_hidden_state[:, 0, :]), dim=-1)

        # 5. Predict Final Labels
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

    @torch.no_grad()
    def generate(self, image, prompt="a picture of", ret_attention=False):
        # 1. Generate Image Embeddings
        image_embeds = self.visual_encoder(image)
        # image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)
        if not self.sample:
            image_embeds_beam = image_embeds.detach().repeat_interleave(self.num_beams, dim=0)
            image_atts_beam = torch.ones(image_embeds_beam.size()[:-1], dtype=torch.long).to(image.device)
            decoder_kwargs = {"encoder_hidden_states": image_embeds_beam, "encoder_attention_mask": image_atts_beam}
        else:
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
            decoder_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask": image_atts}

        # 2. Decode Caption from Image Embeddings
        prompt = [self.prompt] * image.size(0)
        decoder_input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(image.device)
        decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
        # decoder_targets = decoder_input_ids.masked_fill(decoder_input_ids == self.tokenizer.pad_token_id, -100)

        if self.sample:
            # nucleus sampling
            decoder_outputs = self.text_decoder.generate(
                input_ids=decoder_input_ids,
                max_length=self.max_decode_length,
                min_length=self.min_decode_length,
                do_sample=True,
                top_p=self.top_p,
                num_return_sequences=1,
                eos_token_id=self.tokenizer.sep_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                repetition_penalty=1.1,
                output_attentions=True,
                return_dict_in_generate=True,
                **decoder_kwargs,
            )
        else:
            # beam search
            decoder_outputs = self.text_decoder.generate(
                input_ids=decoder_input_ids,
                max_length=self.max_decode_length,
                min_length=self.min_decode_length,
                num_beams=self.num_beams,
                eos_token_id=self.tokenizer.sep_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                repetition_penalty=self.repetition_penalty,
                return_dict_in_generate=True,
                output_attentions=True,
                output_hidden_states=True,
                **decoder_kwargs,
            )
        captions = []
        for output in decoder_outputs.sequences:
            caption = self.tokenizer.decode(output, skip_special_tokens=True)
            captions.append(caption[len(self.prompt) :])
        if ret_attention:
            return captions, decoder_outputs
        else:
            return captions

    def forward(self, image, class_label, ret_caption=False, temperature=None):
        # 1. Generate Image Embeddings
        with self.vision_context():
            image_embeds = self.visual_encoder(image)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
            # image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)

        # 2. Get Pseudo Ground Truth Captions from Momentum Decoder
        with torch.no_grad():
            self._momentum_update()
            image_embeds_m = self.visual_encoder_m(image)
            image_embeds_m = image_embeds_m.detach().repeat_interleave(self.num_beams, dim=0)
            image_atts_m = torch.ones(image_embeds_m.size()[:-1], dtype=torch.long).to(image.device)
            decoder_kwargs_m = {
                "encoder_hidden_states": image_embeds_m,
                "encoder_attention_mask": image_atts_m,
            }
            prompt = [self.prompt] * image.size(0)
            input_ids_m = self.tokenizer(prompt, return_tensors="pt").input_ids.to(image.device)
            input_ids_m[:, 0] = self.tokenizer.bos_token_id
            input_ids_m = input_ids_m[:, :-1]
            # beam search
            decoder_output_m = self.text_decoder_m.generate(
                input_ids=input_ids_m,
                max_length=self.max_decode_length,
                min_length=self.min_decode_length,
                num_beams=self.num_beams,
                eos_token_id=self.tokenizer.sep_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                repetition_penalty=self.repetition_penalty,
                **decoder_kwargs_m,
            )
            captions_m = []
            for output in decoder_output_m:
                caption_m = self.tokenizer.decode(output, skip_special_tokens=True)
                captions_m.append(caption_m[len(self.prompt) :])

        # 3. Decode Caption from Image Embeddings
        with self.caption_context():
            captions_m_text = self.tokenizer(
                captions_m, padding="max_length", truncation=True, max_length=30, return_tensors="pt"
            ).to(image.device)
            decoder_input_ids = captions_m_text.input_ids.clone()
            decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
            decoder_targets = decoder_input_ids.masked_fill(decoder_input_ids == self.tokenizer.pad_token_id, -100)
            decoder_outputs = self.text_decoder(
                decoder_input_ids,
                attention_mask=captions_m_text.attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                labels=decoder_targets,
                return_dict=True,
            )
            tau = self.temperature if temperature is None else temperature
            token_mask = nn.functional.gumbel_softmax(decoder_outputs.logits, tau=tau, hard=True, dim=-1)
            indices = torch.tensor(range(0, token_mask.size(-1))).to(token_mask.device)
            decoded_tokens = (token_mask * indices).sum(-1)
            captions = []
            for output in decoded_tokens.detach():
                caption = self.tokenizer.decode(output, skip_special_tokens=True)
                captions.append(caption[len(self.prompt) :])
            loss_lm = decoder_outputs.loss

        # 4. Encode Caption (and Image Embedding)
        caption_text = self.tokenizer(
            captions,
            padding="max_length",
            truncation=True,
            max_length=decoded_tokens.size(1),
            return_tensors="pt",
        ).to(image.device)
        model_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask": image_atts}
        cross_attn_kwargs = {"mode": "text"} if not self.use_cross_attn else model_kwargs
        caption_text_output = self.text_encoder(
            decoded_tokens,
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


from typing import List


def tie_encoder_decoder_weights(encoder: nn.Module, decoder: nn.Module, base_model_prefix: str, skip_key: str):
    uninitialized_encoder_weights: List[str] = []
    if decoder.__class__ != encoder.__class__:
        logger.info(
            f"{decoder.__class__} and {encoder.__class__} are not equal. In this case make sure that all encoder weights are correctly initialized."
        )

    def tie_encoder_to_decoder_recursively(
        decoder_pointer: nn.Module,
        encoder_pointer: nn.Module,
        module_name: str,
        uninitialized_encoder_weights: List[str],
        skip_key: str,
        depth=0,
    ):
        assert isinstance(decoder_pointer, nn.Module) and isinstance(
            encoder_pointer, nn.Module
        ), f"{decoder_pointer} and {encoder_pointer} have to be of type torch.nn.Module"
        if hasattr(decoder_pointer, "weight") and skip_key not in module_name:
            assert hasattr(encoder_pointer, "weight")
            encoder_pointer.weight = decoder_pointer.weight
            if hasattr(decoder_pointer, "bias"):
                assert hasattr(encoder_pointer, "bias")
                encoder_pointer.bias = decoder_pointer.bias
            print(module_name + " is tied")
            return

        encoder_modules = encoder_pointer._modules
        decoder_modules = decoder_pointer._modules
        if len(decoder_modules) > 0:
            assert (
                len(encoder_modules) > 0
            ), f"Encoder module {encoder_pointer} does not match decoder module {decoder_pointer}"

            all_encoder_weights = set([module_name + "/" + sub_name for sub_name in encoder_modules.keys()])
            encoder_layer_pos = 0
            for name, module in decoder_modules.items():
                if name.isdigit():
                    encoder_name = str(int(name) + encoder_layer_pos)
                    decoder_name = name
                    if not isinstance(decoder_modules[decoder_name], type(encoder_modules[encoder_name])) and len(
                        encoder_modules
                    ) != len(decoder_modules):
                        # this can happen if the name corresponds to the position in a list module list of layers
                        # in this case the decoder has added a cross-attention that the encoder does not have
                        # thus skip this step and subtract one layer pos from encoder
                        encoder_layer_pos -= 1
                        continue
                elif name not in encoder_modules:
                    continue
                elif depth > 500:
                    raise ValueError(
                        "Max depth of recursive function `tie_encoder_to_decoder` reached. It seems that there is a circular dependency between two or more `nn.Modules` of your model."
                    )
                else:
                    decoder_name = encoder_name = name
                tie_encoder_to_decoder_recursively(
                    decoder_modules[decoder_name],
                    encoder_modules[encoder_name],
                    module_name + "/" + name,
                    uninitialized_encoder_weights,
                    skip_key,
                    depth=depth + 1,
                )
                all_encoder_weights.remove(module_name + "/" + encoder_name)

            uninitialized_encoder_weights += list(all_encoder_weights)

    # tie weights recursively
    tie_encoder_to_decoder_recursively(decoder, encoder, base_model_prefix, uninitialized_encoder_weights, skip_key)
    tie_encoder_to_decoder_recursively(decoder, encoder, base_model_prefix, uninitialized_encoder_weights, skip_key)


class BLIP_ViT(BLIP_ExBM):
    def __init__(
        self,
        num_classes,
        med_config="configs/bert_config.json",
        image_size=384,
        vit="base",
        vit_grad_ckpt=False,
        vit_ckpt_layer=0,
        embed_dim=256,
        prompt="a picture of ",
        sample=False,
        num_beams=3,
        max_decode_length=30,
        min_decode_length=10,
        top_p=0.9,
        repetition_penalty=1,
        use_cross_attn=False,
        fixed_caps=False,
        fixed_vit=False,
        tied_encoder=False,
        momentum=1,
        temperature=1,
    ):
        super().__init__(
            num_classes,
            med_config,
            image_size,
            vit,
            vit_grad_ckpt,
            vit_ckpt_layer,
            embed_dim,
            prompt,
            sample,
            num_beams,
            max_decode_length,
            min_decode_length,
            top_p,
            repetition_penalty,
            use_cross_attn,
            fixed_caps,
            fixed_vit,
            tied_encoder,
            momentum,
            temperature,
        )
        self.vision_proj = nn.Linear(self.vision_width, embed_dim)

    @torch.no_grad()
    def infer(self, image, class_label, ret_caption=False):
        # 1. Generate Image Embeddings
        image_embeds = self.visual_encoder(image)

        # Predict Final Labels
        image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)
        logit = self.classifier(image_feat)
        loss_cls = F.cross_entropy(logit, class_label)
        acc = accuracy(logit.detach(), class_label)[0]

        if ret_caption:
            return loss_cls, acc, None
        else:
            return loss_cls, acc

    def forward(self, image, class_label, ret_caption=False, temperature=None):
        # 1. Generate Image Embeddings
        with self.vision_context():
            image_embeds = self.visual_encoder(image)

        # Predict Final Labels
        image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)
        logit = self.classifier(image_feat)
        loss_cls = F.cross_entropy(logit, class_label)
        acc = accuracy(logit.detach(), class_label)[0]

        if ret_caption:
            return loss_cls, torch.tensor(0.0), acc, None
        else:
            return loss_cls, torch.tensor(0.0), acc


def blip_exbm(use_vit_only, **kwargs):
    if use_vit_only:
        model = BLIP_ViT(**kwargs)
    else:
        model = BLIP_ExBM(**kwargs)
    return model
