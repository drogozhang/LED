import math
import torch
import torch.nn as nn


def transform_edges_to_mask(graph_edges, seq_len, symmetry=True):
    bs = graph_edges.size(0)
    sl = seq_len
    # change graph edge [bs,n,2] to 2d index [N, 3]
    batch_idxs = torch.arange(bs).to(
        graph_edges.device).unsqueeze(-1).unsqueeze(-1).expand(bs, graph_edges.size(1), 1)  # bs,n,1
    new_graph_edges = torch.where(
        graph_edges > -1,
        graph_edges,
        graph_edges.new_full(graph_edges.size(), fill_value=sl)
    )
    g_indices = torch.cat((batch_idxs, new_graph_edges), dim=-1).contiguous().view(-1, 3)  # bs*n, 3
    # for _row in range(g_indices.size(0)):
    #     print(g_indices[_row])
    graph_mask = torch.sparse.FloatTensor(  # bs,sl,sl
        g_indices.t(), graph_edges.new_ones([g_indices.size(0)], dtype=graph_edges.dtype),
        torch.Size([bs, sl + 1, sl + 1])
    ).to_dense()[:, :-1, :-1].contiguous()

    if symmetry:
        graph_mask = (graph_mask + torch.transpose(graph_mask, -1, -2)).gt(0).to(graph_mask.dtype)

    return graph_mask  # mask with same type as `graph_edges`


def mask_zero2exp(_mask, dtype):
    _exp_mask = (torch.ones_like(_mask) - _mask).to(dtype) * \
                torch.full([1], fill_value=-10000, dtype=dtype, device=_mask.device)
    return _exp_mask

def exp_mask(_mask, _val, high_rank=False):
    _exp_mask = (torch.ones_like(_mask) - _mask).to(_val.dtype) * \
                torch.full([1], fill_value=-10000, dtype=_val.dtype, device=_val.device)
    if high_rank:
        _exp_mask = _exp_mask.unsqueeze(-1).expand_as(_val)
    return _exp_mask + _val


def zero_mask(_mask, _val, high_rank=False):
    _zero_mask = _mask.to(_val.dtype)
    if high_rank:
        _zero_mask = _zero_mask.unsqueeze(-1).expand_as(_val)
    return _zero_mask * _val


def mask_2d_to_1d(mask_2d, dim=-1, threshold=0):
    return mask_2d.sum(dim).gt(threshold).to(mask_2d.dtype)


def mask_1d_to_2d(mask_1d, other_mask_1d=None):
    if other_mask_1d is None:
        other_mask_1d = mask_1d
    return mask_1d.unsqueeze(-1) * other_mask_1d.unsqueeze(-2)


def prime_to_attn_2d_mask(prime_mask):
    """

    :param prime_mask: [bs,sl] when val>0 is a prime number and val==0 is unmask
    :return:
    """
    bs = prime_mask.size(0)
    sl = prime_mask.size(1)
    mask1d = torch.where(prime_mask > 0, prime_mask, torch.ones_like(prime_mask))
    mask2d_1 = mask1d.unsqueeze(-1).expand(bs, sl, sl)
    mask2d_2 = mask1d.unsqueeze(-2).expand(bs, sl, sl)
    mask2d = (torch.remainder(mask2d_1, mask2d_2) == 0).to(prime_mask.dtype)
    # mask invalid token
    valid = (prime_mask > 0).to(prime_mask.dtype)
    valid2d_1 = valid.unsqueeze(-1).expand(bs, sl, sl)
    valid2d_2 = valid.unsqueeze(-2).expand(bs, sl, sl)

    final_mask2d = (valid2d_1*valid2d_2) * mask2d
    return final_mask2d


def masked_pool(rep_input, rep_mask, high_rank=True, method="mean", return_new_mask=False):

    dim_pool = rep_mask.dim() - 1
    new_mask = (rep_mask.sum(dim=dim_pool) > 0).to(rep_mask.dtype)

    if method == "mean":
        masked_input = zero_mask(rep_mask, rep_input, high_rank=high_rank)
        rep_output = masked_input.sum(dim=dim_pool)
        denominator = rep_mask.to(rep_output.dtype).sum(dim=dim_pool)
        # remove zero
        denominator = torch.where(
            denominator > 0.,
            denominator, torch.full_like(denominator, fill_value=1.)
        )
        if high_rank:
            denominator = denominator.unsqueeze(-1).expand_as(rep_output)
        rep_output /= denominator

    elif method == "max":
        masked_input = exp_mask(rep_mask, rep_input, high_rank=high_rank)
        rep_output = torch.max(masked_input, dim=dim_pool)[0]
    else:
        raise NotImplementedError

    rep_output = zero_mask(new_mask, rep_output, high_rank=high_rank)

    if return_new_mask:
        return rep_output, new_mask
    else:
        return rep_output


def len_mask(_lens, max_len=None):
    max_len = max_len or _lens.max().item()  # []
    rg = torch.arange(0, max_len, dtype=torch.long, device=_lens.device)  # ml
    # expand to [...] + [ml]
    for _ in range(_lens.dim()):
        rg = rg.unsqueeze(0)
    rg = rg.expand(list(_lens.size()) + [max_len])
    expd_lens = _lens.unsqueeze(-1).expand_as(rg)
    return (rg < expd_lens).to(torch.long), max_len


def slice_tensor_v2(rep_input, rep_se):
    """
    :param rep_input: [bs,sl,hn]
    :param rep_se: [bs,nl,2]
    :return:
    """
    bs, sl, hn = rep_input.shape
    _, nl = rep_se.shape[:2]
    device = rep_input.device

    node_lens = rep_se[..., 1] - rep_se[..., 0]  # bs,nl
    node_len_mask, max_node_len = len_mask(node_lens)  # [bs,nl,pl], []
    # refine node_len_mask
    node_len_mask = node_len_mask * ((rep_se[..., 1] - rep_se[..., 0]) > 0).to(torch.long).unsqueeze(-1)  # [bs,nl,pl]

    node_ranges = torch.arange(0, max_node_len, dtype=torch.long, device=device).unsqueeze(
        0).unsqueeze(0).expand([bs, nl, max_node_len])  # bs,nl,pl
    node_indices = (node_ranges + rep_se[..., 0].unsqueeze(-1)) * node_len_mask  # bs,nl,pl
    node_indices = node_indices.contiguous()  # bs,nl,pl
    node_indices_rsp = \
        node_indices.view(bs, nl * max_node_len).unsqueeze(-1).expand(bs, nl * max_node_len, hn)  # bs, nl*pl, hn
    rep_node = torch.gather(rep_input, dim=1, index=node_indices_rsp).view(bs, nl, max_node_len, hn)
    rep_node = zero_mask(node_len_mask, rep_node, high_rank=True)

    return rep_node, node_len_mask  # [bs,nl,pl,hn] & [bs,nl,pl]


def slice_tensor(rep_input, rep_se):
    """

    :param rep_input: [bs,sl,hn]
    :param rep_se: [bs,nl,2]
    :return:
    """
    bs, sl, hn = rep_input.shape
    _, nl = rep_se.shape[:2]
    device = rep_input.device

    node_lens = rep_se[..., 1] - rep_se[..., 0] + 1  # bs,nl
    node_len_mask, max_node_len = len_mask(node_lens)  # [bs,nl,pl], []
    # refine node_len_mask
    node_len_mask = node_len_mask * (rep_se[..., 1] >= 0).to(torch.long).unsqueeze(-1).expand_as(node_len_mask)

    node_ranges = torch.arange(0, max_node_len, dtype=torch.long, device=device).unsqueeze(
        0).unsqueeze(0).expand([bs, nl, max_node_len])  # bs,nl,pl
    node_indices = (node_ranges + rep_se[..., 0].unsqueeze(-1).expand_as(node_ranges)) * node_len_mask
    #    - (1-node_len_mask)  # bs,nl,pl
    node_indices = node_indices.contiguous()
    node_indices_rsp = \
        node_indices.view(bs, nl*max_node_len).unsqueeze(-1).expand(bs, nl*max_node_len, hn) # bs, nl*pl, hn

    rep_node = torch.gather(rep_input, dim=1, index=node_indices_rsp).view(bs, nl, max_node_len, hn)
    rep_node = zero_mask(node_len_mask, rep_node, high_rank=True)

    return rep_node, node_len_mask  # [bs,nl,pl,hn] & [bs,nl,pl]


def slice_tensor_combine_v2(rep_input, rep_se, seq_len):
    """

    :param rep_input:  [bs,nl,pl,hn]
    :param rep_se:  [bs,nl,2]
    :param seq_len: python int
    :return: [bs,sl,hn]
    """
    bs, nl, pl, hn = rep_input.shape
    sl = seq_len
    device = rep_input.device

    # node to token matrix
    rgs = torch.arange(sl, device=device).unsqueeze(0).unsqueeze(0).expand([bs, nl, sl])  # [bs, nl, sl]
    start_indices = rep_se[..., 0].unsqueeze(-1).expand_as(rgs)  # [bs, nl, sl]
    end_indices = rep_se[..., 1].unsqueeze(-1).expand_as(rgs)
    node2tk_mask = (start_indices <= rgs).to(torch.long) * (rgs < end_indices).to(torch.long)  # [bs, nl, sl]

    token_indices = (rgs - start_indices) * node2tk_mask  # [bs, nl, sl]
    token_indices_rsp = token_indices.unsqueeze(-1).expand([bs, nl, sl, hn])  # [bs, nl, sl, hn]

    scatter_org_rep = torch.gather(rep_input, dim=2, index=token_indices_rsp)  # [bs, nl, sl, hn]
    scatter_org_rep = scatter_org_rep * node2tk_mask.to(rep_input.dtype).unsqueeze(-1)  # [bs, nl, sl, hn]

    org_rep = scatter_org_rep.sum(1)  # [bs,sl,hn]

    return org_rep


def reversed_slice_tensor(rep_input, rep_se, seq_len):
    """

    :param rep_input:  [bs,nl,hn]
    :param rep_se:  [bs,nl,2]
    :param seq_len: python int
    :return:
    """
    bs, nl, hn = rep_input.shape
    sl = seq_len
    device = rep_input.device
    # node to token matrix
    rgs = torch.arange(sl, device=device).unsqueeze(0).unsqueeze(0).expand([bs, nl, sl])
    start_indices = rep_se[..., 0].unsqueeze(-1).expand_as(rgs)
    end_indices = rep_se[..., 1].unsqueeze(-1).expand_as(rgs)
    node2tk_mask = (start_indices <= rgs).to(torch.long) * (rgs <= end_indices).to(torch.long)  # [bs, nl, sl]
    tk2node_mask = node2tk_mask.transpose(-1, -2)  # [bs,sl,nl]
    nums_nonzero = tk2node_mask.sum(-1)  # bs,sl
    max_nz = nums_nonzero.max().item()  # max_num_nonzero
    res_mask, _ = len_mask(nums_nonzero, max_nz)  # bs,sl,max_nz
    nums_pad = - nums_nonzero + max_nz  # bs,sl
    pad_mask, max_pd = len_mask(nums_pad)  # bs,sl,max_pd || max_padding
    pad_tk2node_mask = torch.cat([tk2node_mask, pad_mask], dim=-1)  # [bs,sl,nl+max_pd]
    res_indices = pad_tk2node_mask.nonzero()[...,-1].contiguous().view(bs, sl, max_nz)  # [bs*sl*max_nz] -> [bs,sl,max_nz]
    res_indices = zero_mask(res_mask, res_indices)
    # gather
    res_indices_gather = res_indices.view(bs, sl*max_nz).unsqueeze(-1).expand([bs, sl*max_nz, hn])
    res_gather = torch.gather(rep_input, dim=1, index=res_indices_gather).view(bs, sl, max_nz, hn)
    res_gather = zero_mask(res_mask, res_gather, high_rank=True)
    return res_gather, res_mask


# act
def act_name2fn(act_name="linear"):
    if act_name == "linear":
        return lambda x: x
    elif act_name == "relu":
        return torch.relu
    elif act_name == "gelu":
        return gelu
    else:
        KeyError(act_name)


def gelu(x):
    return (
        0.5
        * x
        * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    )


def inverted_softmax(
        query_emb, key_emb,
        query_mask=None, key_mask=None,
        sim_score_fn=None,
        negative_score=False,
):
    if sim_score_fn is None:
        def sim_score_fn(x, y):
            return torch.matmul(x, y.transpose(-1,-2))
    scores = sim_score_fn(query_emb, key_emb)  # [...]+[nq,nk]

    if negative_score:
        scores = - scores

    # softmax along dim=-2
    # mask + softmax
    if query_mask is None:
        sf_scores = torch.softmax(scores, dim=-2)  # [...]+[nq,nk]
    else:
        sf_scores = torch.softmax(exp_mask(query_mask.unsqueeze(-1), scores, high_rank=False), dim=-2)  # [...]+[nq,nk]

    # norm along dim=-1
    if key_mask is None:
        isf_scores = sf_scores
    else:
        isf_scores = zero_mask(key_mask.unsqueeze(-2), sf_scores, high_rank=False)
    isf_scores = isf_scores / (torch.sum(isf_scores, dim=-1, keepdim=True) + 1e-5)

    return isf_scores


def split_dual_inputs_by_token_type(
    input_ids, attention_mask, token_type_ids, cls_token_id=101,
):
    batch_size = input_ids.shape[0]
    device = input_ids.device
    assert token_type_ids is not None and torch.max(token_type_ids).item() == 1

    first_sent_mask = ((1-token_type_ids) * attention_mask == 1).to(torch.long)
    second_sent_mask = (token_type_ids * attention_mask == 1).to(torch.long)

    # first
    first_lens = first_sent_mask.sum(-1)
    first_max_len = first_lens.max().item()
    first_input_ids = input_ids[:, :first_max_len]
    first_attention_mask = len_mask(first_lens, first_max_len)[0]
    first_type_ids = torch.zeros_like(first_attention_mask)

    # second
    second_start_idxs = first_lens.cpu().numpy().tolist()  # bs
    second_lens = second_sent_mask.sum(-1)
    second_max_len = second_lens.max().item()
    def slice_tensor(_tensor):
        _tensor = _tensor.cpu()
        _seq_len = _tensor.shape[1]
        _tensors = []
        for ex_idx ,start_idx in enumerate(second_start_idxs):
            end_idx = start_idx+second_max_len
            if end_idx <= _seq_len:
                curr_ex = _tensor[ex_idx, start_idx:end_idx]
            else:
                curr_ex = torch.cat([
                    _tensor[ex_idx, start_idx:_seq_len], torch.zeros([end_idx-_seq_len], dtype=torch.long)
                ], dim=0)
            _tensors.append(curr_ex)
        return torch.stack(_tensors, dim=0)

    second_input_ids = torch.cat(
        [torch.full([batch_size, 1], fill_value=cls_token_id, dtype=torch.long),
         slice_tensor(input_ids)], dim=1,
    ).contiguous().to(device)
    second_attention_mask = len_mask(second_lens+1, second_max_len+1)[0]
    second_type_ids = torch.ones_like(second_attention_mask)

    return (first_input_ids, first_attention_mask, first_type_ids), \
           (second_input_ids, second_attention_mask, second_type_ids)


def combine_dual_inputs_by_attention_mask(
        sent1_input_ids, sent1_attention_mask, sent2_input_ids, sent2_attention_mask,):
    dtype = sent1_input_ids.dtype
    device = sent1_input_ids.device

    bs, sl1 = sent1_input_ids.shape[:2]
    bs2, sl2 = sent2_input_ids.shape[:2]
    assert bs == bs2

    sent1_lens = sent1_attention_mask.sum(-1)
    sent1_len_list = sent1_lens.cpu().numpy().tolist()

    sent2_lens = sent2_attention_mask.sum(-1)
    sent2_len_list = sent2_lens.cpu().numpy().tolist()

    all_lens = sent1_lens + sent2_lens - 1  # remove the 1 for cls in the second sents
    all_len_list = all_lens.cpu().numpy().tolist()

    all_max_len = max(all_len_list)
    all_attention_mask, _ = len_mask(all_lens, all_max_len)

    res_list = []
    type_list = []
    for idx_ex in range(bs):
        pad_part = torch.zeros([all_max_len - all_len_list[idx_ex],], dtype=dtype, device=device)

        sent1_part, sent2_part = sent1_input_ids[idx_ex, :sent1_len_list[idx_ex]], sent2_input_ids[idx_ex, 1:sent2_len_list[idx_ex]]
        res_list.append(torch.cat([sent1_part, sent2_part, pad_part], dim=0))

        type_list.append(
            torch.cat([
                torch.zeros([sent1_lens[idx_ex], ], dtype=dtype, device=device),
                torch.ones([sent2_lens[idx_ex]-1, ], dtype=dtype, device=device),
                pad_part,
            ], dim=0)
        )

    return torch.stack(res_list, dim=0).contiguous(), all_attention_mask, torch.stack(type_list, dim=0).contiguous()


# ============= masking ============

def mask_out_cls_sep(org_mask):  # int-based mask
    pad_shape = list(org_mask.shape)
    assert pad_shape[-1] >= 2
    pad_shape[-1] = 1
    pad_zeros = torch.zeros(pad_shape, dtype=org_mask.dtype, device=org_mask.device)
    new_mask = torch.cat([pad_zeros, org_mask[..., 2:], pad_zeros], dim=-1).contiguous()
    return new_mask


def mask_token_random(input_ids, attention_mask, prob=0.15, mask_token_id=None):
    assert mask_token_id is not None
    device = input_ids.device

    masked_token_idxs = (torch.rand_like(attention_mask, dtype=torch.float32) < prob).to(torch.long).to(device) * mask_out_cls_sep(attention_mask)
    new_input_ids = input_ids * (1-masked_token_idxs) + mask_token_id * masked_token_idxs
    mlm_labels = input_ids * masked_token_idxs + (-100) * (1-masked_token_idxs)
    return new_input_ids, mlm_labels


def add_prefix(repre_ids, prefix_len=4, prefix_token_id=None, static=True):
    assert prefix_token_id is not None
    device = repre_ids.device
    dtype = repre_ids.dtype
    prefix_shape = list(repre_ids.shape)
    prefix_shape[-1] = prefix_len

    zero_prefix = torch.zeros(prefix_shape, dtype=dtype, device=device)
    if isinstance(prefix_token_id, int):
        if static:
            ids_prefix = zero_prefix + prefix_token_id
        else:
            ids_prefix = zero_prefix + torch.arange(prefix_token_id, (prefix_token_id+prefix_len), dtype=dtype, device=device)
    else:
        ids_prefix = zero_prefix + torch.tensor(prefix_token_id, dtype=dtype, device=device)

    new_repre_ids = torch.cat(
        [
            repre_ids[..., :1],
            ids_prefix,
            repre_ids[..., 1:],
        ], dim=-1).contiguous()
    return new_repre_ids


def gen_special_self_attn_mask(
        seq_len, uni_direction=False, single_side_size=None, seq_mask=None,
        device=None,
):
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

    mask = mask.to(device)  # [sl,sl]
    if seq_mask is not None:
        mask = mask.unsqueeze(0) * seq_mask.unsqueeze(-2)  # [1,sl,sl] * [bs,1,sl]

    return mask  # return [seq_len, seq_len] or [bs,sl,sl]



# def add_special_prefix(input_ids, attention_mask, prefix_len=4, special_token_id=None, static_id=True):
#     assert special_token_id is not None
#     device = input_ids.device
#     # assert input_ids.ndim == 2 and attention_mask.ndim == 2
#     prefix_shape = list(input_ids.shape)
#     prefix_shape[-1] = prefix_len
#
#     attn_prefix = torch.ones(prefix_shape, dtype=attention_mask.dtype, device=device)
#     if static_id:
#         ids_prefix = special_token_id * attn_prefix
#     else:
#         ids_prefix = torch.zeros(prefix_shape, dtype=attention_mask.dtype, device=device) + \
#                      torch.arange(special_token_id, (special_token_id+prefix_len),
#                                   dtype=attention_mask.dtype, device=device)
#
#     new_input_ids = torch.cat(
#         [
#             input_ids[..., :1],
#             ids_prefix,
#             input_ids[..., 1:],
#         ], dim=-1).contiguous()
#
#     new_attention_mask = torch.cat(
#         [
#             attn_prefix,
#             attention_mask,
#         ], dim=-1).contiguous()
#
#     pad_attention_mask = torch.cat(
#         [
#             attention_mask[..., :1],
#             torch.zeros_like(attn_prefix),
#             attention_mask[..., 1:],
#         ], dim=-1).contiguous()
#
#     return new_input_ids, new_attention_mask, pad_attention_mask


