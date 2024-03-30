import torch
import math
import copy
from torch import nn
from torch.nn import functional as F


def clones(module, n):
    """
    Produce n identical layers.
    :param module: nn.Module
    :param n: int
    :return: torch.nn.ModuleList
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def linear(input_size, output_size, bias=True, dropout: float = None):
    lin = nn.Linear(input_size, output_size, bias=bias)
    if dropout is not None:
        return nn.Sequential(nn.Dropout(dropout), lin)
    else:
        return lin


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm
    """
    def __init__(self, size, dropout, residual_connection, use_LayerNorm):
        super(SublayerConnection, self).__init__()
        self.residual_connection = residual_connection
        self.use_LayerNorm = use_LayerNorm
        self.dropout = nn.Dropout(dropout)
        if self.use_LayerNorm:
            self.norm = nn.LayerNorm(size)

    def forward(self, x, sublayer):
        """
        :param x: (batch, T, d_model)
        :param sublayer: nn.Module
        :return: (batch, T, d_model)
        """
        if self.residual_connection and self.use_LayerNorm:
            return x + self.dropout(sublayer(self.norm(x)))
        if self.residual_connection and (not self.use_LayerNorm):
            return x + self.dropout(sublayer(x))
        if (not self.residual_connection) and self.use_LayerNorm:
            return self.dropout(sublayer(self.norm(x)))


def attention(query, key, value, mask=None, dropout=None):
    """
    :param query: (batch, h, T1, d_k)
    :param key: (batch, h, T2, d_k)
    :param value: (batch, h, T2, d_k)
    :param mask: (batch, 1, T2, T2)
    :param dropout:
    :return: (batch, h, T1, d_k), (batch, h, T1, T2)
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # scores: (batch, h, T1, T2)

    if mask is not None:
        scores = scores.masked_fill_(mask == 0, -1e9)  # -1e9 means attention scores=0
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    # p_attn: (batch, h, T1, T2)

    return torch.matmul(p_attn, value), p_attn  # (batch, h, T1, d_k), (batch, h, T1, T2)


class MultiHeadAttention(nn.Module):
    def __init__(self, nb_head, d_model, dropout=.0):
        super(MultiHeadAttention, self).__init__()
        assert d_model % nb_head == 0
        self.d_k = d_model // nb_head
        self.h = nb_head
        self.lin = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """
        :param query: (batch, T, d_model)
        :param key: (batch, T, d_model)
        :param value: (batch, T, d_model)
        :param mask: (batch, T, T)
        :return: x: (batch, T, d_model)
        """
        if mask is not None:
            mask = mask.unsqueeze(1)  # (batch, 1, T, T), same mask applied to all h heads.

        nb_batch = query.size(0)

        # (batch, T, d_model) -view-> (batch, T, h, d_k) -transpose(2,3)-> (batch, h, T, d_k)
        query, key, value = [x.view(nb_batch, -1, self.h, self.d_k).transpose(2, 3) for x in (query, key, value)]

        # apply attention on all the projected vectors in batch
        x, attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # x:(batch, h, T1, d_k)
        # attn:(batch, h, T1, T2)

        x = x.transpose(2, 3).contiguous()  # (batch, T1, h, d_k)
        x = x.view(nb_batch, -1, self.h * self.d_k)  # (batch, T1, d_model)
        return self.lin(x)


# granularity feature extraction module
class gfem(nn.Module):
    def __init__(self, nb_head, d_model, t_train, t_pred, dropout=.0, residual_connection=True, use_layernorm=True):
        super(gfem, self).__init__()
        self.attn = MultiHeadAttention(nb_head, d_model)
        self.feed_forward = nn.Linear(d_model, d_model)

        self.sublayer = clones(SublayerConnection(d_model, dropout, residual_connection, use_layernorm), 2)
        self.residual_connection = residual_connection
        self.use_layernorm = use_layernorm
        self.size = d_model

        self.pred = nn.Linear(t_train, t_pred)
        self.rec = nn.Linear(t_train, t_train)

    def forward(self, x):
        """
        :param x: (batch_size, t_train, d_model)
        :return:
            (batch_size, t_train, d_model)
            (batch_size, t_pred, d_model)
        """
        if self.residual_connection or self.use_LayerNorm:
            x = self.sublayer[0](x, lambda x: self.attn(x, x, x))
            x = self.sublayer[1](x, self.feed_forward)
        else:
            x = self.attn(x, x, x)
            x = self.feed_forward(x)

        r, g = self.pred(x.transpose(-1, -2)).transpose(-1, -2), self.rec(x.transpose(-1, -2)).transpose(-1, -2)
        return r, g


# multi-granularity residual neural network
class mrnn(nn.Module):
    def __init__(self, gfs, src_dense, align, intervals, generator):
        super(mrnn, self).__init__()
        self.intervals = intervals
        self.gfs = gfs
        self.dense = src_dense
        self.alignment = align
        self.predict_generator = generator

    def forward(self, x):
        """
        :param x: list, Each position of the list represent a granularity data
        :return: (b, t_pred, f)
        """
        assert len(x) == len(self.intervals)

        # (b, t[i], f) -> (b, t, f) -> (b, t, d_model)
        for idx, block in enumerate(self.alignment):
            x[idx] = block(x[idx].transpose(-1, -2)).transpose(-1, -2)
            x[idx] = self.dense(x[idx])

        y_pred, p, refactor = [], x[0], []
        for idx, block in enumerate(self.gfs):
            r, g = block(p)
            y_pred.append(r)
            refactor.append(g)
            if idx != len(self.intervals) - 1:
                p = x[idx + 1] - sum(refactor)

        return self.predict_generator(sum(y_pred))


def make_model(feature_size, num_for_train, num_for_predict, intervals, d_model, nb_heads):
    c = copy.deepcopy

    src_dense = nn.Linear(feature_size, d_model)

    gf = gfem(nb_heads, d_model, num_for_train, num_for_predict)

    gfs = clones(gf, len(intervals))

    align = []
    for idx in range(len(intervals)):
        al = nn.Linear(intervals[idx] * num_for_train, num_for_train)
        align.append(al)

    alignment = nn.ModuleList(al for al in align)

    generator = nn.Linear(d_model, feature_size)

    model = mrnn(gfs, src_dense, alignment, intervals, generator)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model
