import torch


def stp(s: torch.Tensor, ts: torch.Tensor):  # scalar tensor product
    extra_dims = (1,) * (ts.dim() - 1)
    return s.view(-1, *extra_dims).type_as(ts) * ts


