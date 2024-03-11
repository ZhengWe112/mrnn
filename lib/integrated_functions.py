import torch


def growth_rate(tensor, interval):
    """
    Integrating coarse-grained data into fine-grained data
    :param tensor: shape is (batch_size, timestep*interval, 1)
    :param interval: the number of fine-grained data in one timestep
    :return:
        tensor: shape is (batch_size, timestep, 1)
    """
    origin = tensor.split(interval, dim=1)
    res = []
    for t in origin:
        t = t + 1
        t = torch.prod(t, dim=1, keepdim=True)
        t = t - 1
        res.append(t)

    return torch.cat(res, dim=1)


def max_value(tensor, interval):
    origin = tensor.split(interval, dim=1)
    res = []
    for t in origin:
        t = torch.max(t, dim=1, keepdim=True)
        res.append(t)

    return torch.cat(res, dim=1)
