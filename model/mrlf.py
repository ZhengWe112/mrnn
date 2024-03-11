import torch
from torch import nn
from torch.nn import functional as F
from lib.utils import frobenius_norm


# granularity feature extraction module
class gfem(nn.Module):
    def __init__(self, feature_size, hidden_size, pred_size, train_size, num_layers=2):
        super(gfem, self).__init__()
        self.enc = nn.GRU(feature_size, hidden_size, num_layers, batch_first=True)
        self.pred = nn.Linear(train_size, pred_size)
        self.rec = nn.Linear(train_size, train_size)

        self.ar = nn.Linear(train_size - 1, 1)
        self.w = nn.Linear(feature_size, feature_size)

    def forward(self, x):
        # x, hn = self.enc(x.transpose(1, 2))
        # x = torch.tanh(x.transpose(1, 2))
        r = torch.tanh(self.pred(x))
        g = torch.tanh(self.rec(x))

        c = self.ar(x[:, :, :-1])
        alpha = torch.exp(self.w(x[:, :, -1].unsqueeze(1).transpose(1, 2)) @ c)
        return r, g, alpha


# multi-granularity residual learning framework
class mrlf(nn.Module):
    def __init__(self, feature_size, hidden_size, num_for_train, num_for_predict, intervals):
        super(mrlf, self).__init__()
        self.intervals = intervals

        basic_block, align = [], []
        for idx in range(len(intervals)):
            pred_size = num_for_predict
            train_size = num_for_train
            bl = gfem(feature_size, hidden_size, pred_size, train_size)
            basic_block.append(bl)
            al = nn.Linear(intervals[idx] * num_for_train, num_for_train)
            align.append(al)

        self.alignment = nn.ModuleList(
            al for al in align
        )

        self.basic_block = nn.ModuleList(
            bl for bl in basic_block
        )

    def forward(self, x):
        assert len(x) == len(self.intervals)

        for idx, block in enumerate(self.alignment):
            x[idx] = block(x[idx])

        res, weights, p, sublosslist = [], [], x[0], []
        for idx, block in enumerate(self.basic_block):
            r, g, a = block(p)
            if idx != len(self.intervals) - 1:
                p = x[idx+1] - g
                sublosslist.append(frobenius_norm(p))
            res.append(r)
            weights.append(a)

        subloss = torch.cat(sublosslist, dim=0)
        subloss = torch.sum(subloss, dim=0)

        res_tensor = torch.cat(res, dim=2)
        weights_tensor = torch.cat(weights, dim=2)
        weights_tensor = F.softmax(weights_tensor, dim=2)
        return (res_tensor * weights_tensor).sum(-1, keepdim=True), subloss
