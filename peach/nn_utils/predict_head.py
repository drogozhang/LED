import torch
import torch.nn as nn
from peach.nn_utils.nn import LinearWithAct


class MultiClsHead(nn.Module):
    def __init__(self, num_in_features, num_cls, dropout_prob=None):
        super(MultiClsHead, self).__init__()

        # self.fns = nn.Sequential(
        #     nn.Dropout(dropout_prob) if (num_hidden is not None and dropout_prob is not None) else nn.Sequential(),
        #     nn.Linear(num_in_features, num_hidden) if num_hidden is not None else nn.Sequential(), )

        self.ms = nn.Sequential(
            nn.Dropout(dropout_prob) if dropout_prob is not None else nn.Sequential(),
            nn.Linear(num_in_features, num_cls, bias=True)
        )

    def forward(self, hidden_states, labels=None, reduction="mean", *args, **kwargs):
        return_list = []

        logits = self.ms(hidden_states)
        return_list.append(logits)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(reduction=reduction)
            loss = loss_fn(logits, labels)
            return_list.insert(0, loss)
        return tuple(return_list)


class CosSimHead(nn.Module):
    def __init__(self, hidden_size=None, dropout_prob=0., act_name="gelu"):
        super(CosSimHead, self).__init__()
        if hidden_size is not None:
            self.proj = LinearWithAct(hidden_size, hidden_size, dropout_prob, act_name)
        else:
            self.proj = None

    def forward(self, rep_sent1, rep_sent2, labels=None, reduction="mean", use_proj=True, *args, **kwargs):
        if self.proj is not None and use_proj:
            rep_sent1 = self.proj(rep_sent1)
            rep_sent2 = self.proj(rep_sent2)

        return_list = []

        sim_scores = (torch.cosine_similarity(rep_sent1, rep_sent2, dim=-1) + 1.) / 2

        return_list.append(sim_scores)

        if labels is not None:
            loss_fn = nn.MSELoss(reduction=reduction)
            loss = loss_fn(sim_scores, labels)
            return_list.insert(0, loss)
        return tuple(return_list)


class MlpSimHead(nn.Module):
    def __init__(self, hidden_size, num_cls, dropout_prob=None, symmetry=True, use_extra_hidden=False, act_name="gelu"):
        super(MlpSimHead, self).__init__()
        if use_extra_hidden:
            self.proj = LinearWithAct(hidden_size, hidden_size, dropout_prob, act_name)
        else:
            self.proj = None

        num_in_features = 2 * hidden_size if symmetry else 4 * hidden_size
        self.multi_cls_head = MultiClsHead(num_in_features, num_cls, dropout_prob)

        self.symmetry = symmetry

    def forward(self, rep_sent1, rep_sent2, labels=None, reduction="mean", use_proj=True, *args, **kwargs):
        if self.proj is not None and use_proj:
            rep_sent1 = self.proj(rep_sent1)
            rep_sent2 = self.proj(rep_sent2)

        if self.symmetry:
            features = torch.cat(
                [torch.abs(rep_sent1-rep_sent2), rep_sent1 * rep_sent2], dim=-1).contiguous()
        else:
            features = torch.cat([
                rep_sent1, rep_sent2,
                rep_sent1 - rep_sent2,
                rep_sent1 * rep_sent2], dim=-1
            ).contiguous()

        return self.multi_cls_head(features, labels, reduction)

