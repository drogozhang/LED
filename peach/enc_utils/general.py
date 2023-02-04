import torch
import torch.nn as nn
import torch.distributed as dist

MAX_TITLE_LENGTH = 32
MAX_QUERY_LENGTH = 32
MAX_SENT_LENGTH = 64

def get_representation_tensor(model_output):  # for compatibility with huggingface models
    if isinstance(model_output, dict):
        if "sentence_embedding" in model_output:
            return model_output["sentence_embedding"].contiguous()
        elif "last_hidden_state" in model_output:  # [CLS] embedding
            return model_output["last_hidden_state"][..., 0, :].contiguous()
        else:
            raise AttributeError(model_output.keys())
    else:
        return model_output


def preproc_inputs(
        input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
        input_ids_query=None, attention_mask_query=None, token_type_ids_query=None, position_ids_query=None, 
):
    org_doc_shape = input_ids.shape
    org_query_shape = input_ids_query.shape if input_ids_query is not None else None
    if input_ids.ndim >= 3:
        assert input_ids.ndim == 3
        high_dim_flag = True
        num_docs = input_ids.shape[1]

        input_ids = input_ids.view(-1, input_ids.shape[-1])
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, position_ids.shape[-1])

        if input_ids_query is not None:
            input_ids_query = input_ids_query.view(-1, input_ids_query.shape[-1])
            attention_mask_query = attention_mask_query.view(-1, attention_mask_query.shape[-1])

            if token_type_ids_query is not None:
                token_type_ids_query = token_type_ids_query.view(-1, token_type_ids_query.shape[-1])
            if position_ids_query is not None:
                position_ids_query = position_ids_query.view(-1, position_ids_query.shape[-1])
    else:
        high_dim_flag = False
        num_docs = 1
    return (high_dim_flag, num_docs), (org_doc_shape, org_query_shape), \
           (input_ids, attention_mask, token_type_ids, position_ids,), \
           (input_ids_query, attention_mask_query, token_type_ids_query, position_ids_query,)


def gen_special_self_attn_mask(seq_len, uni_direction=False, single_side_size=None, seq_mask=None):
    idx_seq = torch.arange(seq_len, dtype=torch.long)
    query = idx_seq.unsqueeze(-1).tile([1, seq_len])
    key = idx_seq.unsqueeze(-2).tile([seq_len, 1])

    if uni_direction:
        mask = (key <= query).to(torch.long)
    else:
        mask = torch.ones([seq_len, seq_len], dtype=torch.long)

    if single_side_size is not None:
        win_mask = torch.logical_and(
            key <= query + single_side_size, key >= query - single_side_size,
        ).to(torch.long)
        mask = mask * win_mask

    if seq_mask is not None:
        mask = mask * seq_mask.unsqeeze(-2)

    return mask  # return [seq_len, seq_len]

