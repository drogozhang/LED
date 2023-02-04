import torch
import torch.nn as nn
from peach.nn_utils.general import *
from transformers.activations import ACT2FN

class BilinearCompFn(nn.Module):
    def __init__(self, in1_features, in2_features, rescale=True):
        super(BilinearCompFn, self).__init__()
        self._rescale = rescale
        self._linear = nn.Linear(in1_features, in2_features, bias=False)

    def forward(self, input1, input2):
        trans_input1 = self._linear(input1)  # [bs,sl,hn2]
        attn_scores = torch.bmm(  # bs,sl,sl
            trans_input1, torch.transpose(input2, -2, -1))
        if self._rescale:
            average_len = 0.5 * (input1.size(-2) + input2.size(-2))
            rescale_factor = average_len ** 0.5
            attn_scores /= rescale_factor
        return attn_scores


class BilinearAttn(nn.Module):
    def __init__(self, hn_q, hn_k):
        super(BilinearAttn, self).__init__()

        self._attn_comp = BilinearCompFn(hn_q, hn_k, rescale=True)
        self._attn_softmax = torch.nn.Softmax(dim=-1)
        
    def forward(self, q, k=None, v=None, attn_mask=None, **kwargs):
        assert attn_mask is not None
        k = q if k is None else k
        v = q if v is None else v

        g_attn_scores = self._attn_comp(q, k)  # bs,sl1,sl2
        g_attn_prob = self._attn_softmax(exp_mask(attn_mask, g_attn_scores))  # bs,sl1,sl2
        g_attn_res = torch.bmm(g_attn_prob, v)  # [bs,sl1,sl2]x[bs,sl2,hn] ==> [bs,sl,hn]
        g_attn_res = zero_mask(mask_2d_to_1d(attn_mask), g_attn_res, high_rank=True)
        return g_attn_res


class LinearWithAct(nn.Module):
    def __init__(self, in_features, out_features, dropout_prob=0., act_name=None):
        super(LinearWithAct, self).__init__()
        self.dp = nn.Dropout(dropout_prob)
        self.proj = nn.Linear(in_features, out_features)
        self.act_name = act_name

    def forward(self, hidden_states, *args, **kwargs):
        x = self.dp(hidden_states)
        x = self.proj(x)
        if self.act_name is not None:
            x = act_name2fn(self.act_name)(x)
        return x


class TwoLayerMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_act, middle_dim=None, drop_prob=0.1):
        super(TwoLayerMLP, self).__init__()
        middle_dim = middle_dim or in_dim
        self.dense = nn.Linear(in_dim, middle_dim)
        self.dropout = nn.Dropout(drop_prob)
        self.out_proj = nn.Linear(middle_dim, out_dim)

        self.act_fn = ACT2FN[hidden_act]

    def forward(self, features, **kwargs):
        x = features  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class SimpleClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_labels=None):  # with_pool=True,
        super().__init__()
        self.num_labels = num_labels
        self.score = nn.Linear(hidden_size, self.num_labels, bias=False)

    def forward(self, hidden_states, **kwargs):
        logits = self.score(hidden_states)
        return logits


class AttentionPooler(nn.Module):
    def __init__(self, in_dim, hidden_act, drop_prob, use_pool_as_feature=False):
        super().__init__()
        self.use_pool_as_feature = use_pool_as_feature
        self.attn_mlp = TwoLayerMLP(
            2 * in_dim if use_pool_as_feature else in_dim,
            1, hidden_act, middle_dim=in_dim, drop_prob=drop_prob)
        # self.softmax_fn = nn.Softmax(dim=-1)

    def forward(self, hidden_states, attention_mask=None, *args, **kwargs):
        if self.use_pool_as_feature:
            pooled_feature = masked_pool(
                hidden_states, hidden_states, high_rank=True, return_new_mask=False)
            pooled_feature = pooled_feature.unsqueeze(-2).expand_as(hidden_states)
            hidden_states = torch.cat([hidden_states, pooled_feature], dim=-1)

        attn_scores = self.attn_mlp(hidden_states).squeeze(-1)
        attn_scores = exp_mask(attention_mask, attn_scores)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        results = torch.matmul(attn_probs.unsqueeze(-2), hidden_states).squeeze(-2)
        return results


class MyFeedForwardModule(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_act, layer_norm_eps, hidden_dropout_prob):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        if isinstance(hidden_act, str):
            self.intermediate_act_fn = ACT2FN[hidden_act]
        else:
            self.intermediate_act_fn = hidden_act

        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, *arg, **kwargs):
        input_tensor = hidden_states

        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


