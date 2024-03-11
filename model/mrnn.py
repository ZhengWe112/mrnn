import torch
from torch import nn
from torch.nn import functional as F


def linear(input_size, output_size, bias=True, dropout: float = None):
    lin = nn.Linear(input_size, output_size, bias=bias)
    if dropout is not None:
        return nn.Sequential(nn.Dropout(dropout), lin)
    else:
        return lin


# granularity feature extraction module
class gfem(nn.Module):
    def __init__(self, units, pred_size, train_size, num_block_layers=4, dropout=0.1, share_thetas=False):
        super(gfem, self).__init__()

        fc_stack = [
            nn.Linear(train_size, units),
            nn.ReLU(),
        ]
        for _ in range(num_block_layers - 1):
            fc_stack.extend([linear(units, units, dropout=dropout), nn.ReLU()])
        self.fc = nn.Sequential(*fc_stack)

        if share_thetas:
            self.theta_p_fc = self.theta_r_fc = nn.Linear(units, units, bias=False)
        else:
            self.theta_p_fc = nn.Linear(units, units, bias=False)
            self.theta_r_fc = nn.Linear(units, units, bias=False)

        self.pred = nn.Linear(units, pred_size)
        self.rec = nn.Linear(units, train_size)

    def forward(self, x):
        x = self.fc(x)

        theta_p = F.relu(self.theta_p_fc(x))
        theta_r = F.relu(self.theta_r_fc(x))

        return self.pred(theta_p), self.rec(theta_r)


# multi-granularity residual neural network
class mrnn(nn.Module):
    def __init__(self, feature_size, units, num_for_train, num_for_predict, intervals):
        super(mrnn, self).__init__()
        self.intervals = intervals

        basic_block, align = [], []
        for idx in range(len(intervals)):
            pred_size = num_for_predict
            train_size = num_for_train
            bl = gfem(units, pred_size, train_size)
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

        y_pred, p, refactor = [], x[0], []
        for idx, block in enumerate(self.basic_block):
            r, g = block(p)
            y_pred.append(r)
            refactor.append(g)
            if idx != len(self.intervals) - 1:
                p = x[idx + 1] - sum(refactor)

        return sum(y_pred)
