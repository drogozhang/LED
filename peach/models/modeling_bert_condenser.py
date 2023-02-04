from transformers.models.bert.modeling_bert import *


class BertForCondenser(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)

        # condenser add:
        self.n_head_layers = getattr(config, "n_head_layers", 2)
        self.skip_from = getattr(config, "skip_from", 6)
        self.c_head = nn.ModuleList(
            [BertLayer(self.config) for _ in range(self.n_head_layers)])

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=True,  # force get all hidden states
            return_dict=True,  # force return_dict
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        # condenser structure
        attention_mask_etd = self.get_extended_attention_mask(
            attention_mask, attention_mask.shape, attention_mask.device)

        head_hidden_states = [
            torch.cat([  # combine middle cls and low hiddens
                sequence_output[:, :1],  # cls_hiddens
                outputs["hidden_states"][self.skip_from][:, 1:], # skip_hiddens
            ], dim=1).contiguous(), ]
        for layer in self.c_head:
            layer_out = layer(
                head_hidden_states[-1],
                attention_mask_etd,
            )
            head_hidden_states.append(layer_out[0])
        head_prediction_scores = self.cls(head_hidden_states[-1])

        masked_lm_loss, head_masked_lm_loss = None, None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            # loss 1: middle layer (original)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            # loss 2: high layer
            head_masked_lm_loss = loss_fct(
                head_prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores, head_prediction_scores) + outputs[2:] + (head_hidden_states, )
            return ((masked_lm_loss, head_masked_lm_loss) + output) if masked_lm_loss is not None else output

        return_dict = MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        # add condenser's outputs
        return_dict["head_loss"] = head_masked_lm_loss
        return_dict["head_logits"] = head_prediction_scores
        return_dict["head_hidden_states"] = head_hidden_states

        return return_dict

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        #  add a dummy token
        assert self.config.pad_token_id is not None, "The PAD token should be defined for generation"
        attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
        dummy_token = torch.full(
            (effective_batch_size, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device
        )
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        return {"input_ids": input_ids, "attention_mask": attention_mask}