import torch
import torch.nn as nn
from peach.models.modeling_bert_condenser import BertForCondenser
from transformers.models.bert.modeling_bert import BertForMaskedLM
from transformers.models.roberta.modeling_roberta import RobertaForMaskedLM

from transformers.models.distilbert.modeling_distilbert import DistilBertForMaskedLM

from peach.nn_utils.general import len_mask, mask_out_cls_sep, add_prefix


def add_model_hyperparameters(parser):
    # applicable to all
    parser.add_argument("--keep_special_tokens", action="store_true")

    # condenser-specific
    parser.add_argument("--condenser_pattern", type=str, default="main", choices=[
        "main", "head", "mean", "max", "cat", ])

    return [
        "keep_special_tokens",
        "condenser_pattern",
    ]


class SpladePooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, token_embeddings, attention_mask, **kwargs):
        saturated_token_embeddings = torch.log(1. + torch.relu(token_embeddings)) * attention_mask.unsqueeze(-1)
        return_dict = {
            # "token_embeddings": token_embeddings,
            # "attention_mask": attention_mask,
            "saturated_token_embeddings": saturated_token_embeddings,
        }
        sentence_embedding = torch.max(saturated_token_embeddings, dim=1).values
        return_dict["sentence_embedding"] = sentence_embedding
        return return_dict


class ConSpladeEnocder(BertForCondenser):
    def __init__(self, config):
        super().__init__(config)
        self.spalde_pooler = SpladePooler(config)

    @property
    def embedding_dim(self):
        vocab_size = self.config.vocab_size
        if self.config.condenser_pattern == 'cat':
            return vocab_size * 2
        else:
            return vocab_size

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            # head_mask=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None,
            # labels=None, output_attentions=None, output_hidden_states=None,
            return_dict=None,
            **kwargs,
    ):
        encoder_outputs = super().forward(
            input_ids=input_ids, attention_mask=attention_mask,
            labels=None, output_hidden_states=True) #  token_type_ids, position_ids

        main_pooling_out_dict = self.spalde_pooler(
            token_embeddings=encoder_outputs.logits,
            attention_mask=attention_mask if self.config.keep_special_tokens else mask_out_cls_sep(attention_mask),
        )

        head_pooling_out_dict = self.spalde_pooler(
            token_embeddings=encoder_outputs["head_logits"],
            attention_mask=attention_mask if self.config.keep_special_tokens else mask_out_cls_sep(attention_mask),
        )

        # comb
        condenser_pattern = self.config.condenser_pattern
        main_emb, head_emb = main_pooling_out_dict["sentence_embedding"], head_pooling_out_dict["sentence_embedding"]
        if condenser_pattern == "main":
            sent_emb = main_emb
        elif condenser_pattern == "head":
            sent_emb = head_emb
        elif condenser_pattern == "mean":
            sent_emb = torch.stack([main_emb, head_emb], dim=-2).mean(-2)
        elif condenser_pattern == "max":
            sent_emb = torch.stack([main_emb, head_emb], dim=-2).max(-2).values
        elif condenser_pattern == "cat":
            sent_emb = torch.cat([main_emb, head_emb], dim=-1)
        else:
            raise NotImplementedError(condenser_pattern)

        # data to return
        dict_for_return = {
            "hidden_states": encoder_outputs.hidden_states[-1],
            "prediction_logits": encoder_outputs.logits,
            "head_prediction_logits": encoder_outputs["head_logits"],
            "sentence_embedding": sent_emb,
        }

        main_pooling_out_dict = dict(("main_" + k, main_pooling_out_dict[k]) for k in main_pooling_out_dict)
        dict_for_return.update(main_pooling_out_dict)
        head_pooling_out_dict = dict(("head_" + k, head_pooling_out_dict[k]) for k in head_pooling_out_dict)
        dict_for_return.update(head_pooling_out_dict)

        return dict_for_return if return_dict else dict_for_return["sentence_embedding"]


class BertSpladeEnocder(BertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        self.spalde_pooler = SpladePooler(config)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            # head_mask=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None,
            # labels=None, output_attentions=None, output_hidden_states=None,
            return_dict=None,
            **kwargs,
    ):
        encoder_outputs = super().forward(
            input_ids=input_ids, attention_mask=attention_mask,
            labels=None, output_hidden_states=True) #  token_type_ids, position_ids

        pooling_out_dict = self.spalde_pooler(
            token_embeddings=encoder_outputs.logits,
            attention_mask=attention_mask if self.config.keep_special_tokens else mask_out_cls_sep(attention_mask),)

        # data to return
        dict_for_return = {
            "hidden_states": encoder_outputs.hidden_states[-1],
            "all_hidden_states": encoder_outputs.hidden_states,
            "prediction_logits": encoder_outputs.logits,
        }
        dict_for_return.update(pooling_out_dict)

        return dict_for_return if return_dict else dict_for_return["sentence_embedding"]

class RobertaSpladeEnocder(RobertaForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        self.spalde_pooler = SpladePooler(config)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            # head_mask=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None,
            # labels=None, output_attentions=None, output_hidden_states=None,
            return_dict=None,
            **kwargs,
    ):
        encoder_outputs = super().forward(
            input_ids=input_ids, attention_mask=attention_mask,
            labels=None, output_hidden_states=True) #  token_type_ids, position_ids

        pooling_out_dict = self.spalde_pooler(
            token_embeddings=encoder_outputs.logits,
            attention_mask=attention_mask if self.config.keep_special_tokens else mask_out_cls_sep(attention_mask),)

        # data to return
        dict_for_return = {
            "hidden_states": encoder_outputs.hidden_states[-1],
            "prediction_logits": encoder_outputs.logits,
        }
        dict_for_return.update(pooling_out_dict)

        return dict_for_return if return_dict else dict_for_return["sentence_embedding"]

class DistilBertSpladeEnocder(DistilBertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        self.spalde_pooler = SpladePooler(config)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            attention_mask_3d=None,
            # head_mask=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None,
            # labels=None, output_attentions=None, output_hidden_states=None,
            return_dict=None,
            **kwargs,
    ):
        encoder_outputs = super().forward(
            input_ids=input_ids, attention_mask=attention_mask_3d if attention_mask_3d is not None else attention_mask,
            labels=None, output_hidden_states=True) #  token_type_ids, position_ids

        pooling_out_dict = self.spalde_pooler(
            token_embeddings=encoder_outputs.logits,
            attention_mask=attention_mask if self.config.keep_special_tokens else mask_out_cls_sep(attention_mask),)

        # data to return
        dict_for_return = {
            "hidden_states": encoder_outputs.hidden_states[-1],
            "all_hidden_states": encoder_outputs.hidden_states,
            "prediction_logits": encoder_outputs.logits,
        }
        dict_for_return.update(pooling_out_dict)

        return dict_for_return if return_dict else dict_for_return["sentence_embedding"]


