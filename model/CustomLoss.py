import torch
from torch import nn
from torch.nn import functional as F


class CustomLoss(nn.Module):
    def __init__(self, device):
        super(CustomLoss, self).__init__()
        self.DEVICE = device

    def forward(self, pred, target, recloss):
        mse_per_slice = []
        for i in range(pred.shape[0]):
            slice1 = pred[i, :, :]
            slice2 = target[i, :, :]
            mse = ((slice1 - slice2) ** 2).mean()
            mse_per_slice.append(mse)

        mse_per_slice_tensor = torch.tensor(mse_per_slice).to(self.DEVICE)
        return torch.mean(recloss + mse_per_slice_tensor)
