import torch
import torch.nn as nn


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, metric=None, temp=None, ):
        super().__init__()

        metric = metric if metric is not None else "cos"
        temp = temp if temp is not None else 0.05

        assert metric in ["cos", "dot"]
        self.metric = metric
        if metric == "cos":
            self.temp = temp
            self.cos = nn.CosineSimilarity(dim=-1)
        elif metric == "dot":
            pass
        else:
            raise NotImplementedError

    def forward(self, x, y):
        if self.metric == "cos":
            return self.cos(x.unsqueeze(-2), y.unsqueeze(-3)) / self.temp
        elif self.metric == "dot":
            return torch.matmul(x, y.transpose(-2, -1))
        else:
            raise NotImplementedError

    def forward_qd_pair(self, x, y):
        if self.metric == "cos":
            return self.cos(x, y) / self.temp
        elif self.metric == "dot":
            return (x * y).sum(-1)
        else:
            raise NotImplementedError