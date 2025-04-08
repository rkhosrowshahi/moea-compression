import logging
from typing import Tuple

import torch
from torch import nn
from torch.nn.utils.prune import BasePruningMethod

log = logging.getLogger(__name__)

class RangePruning(BasePruningMethod):
    PRUNING_TYPE = 'unstructured'  # This is unstructured pruning

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def compute_mask(self, tensor, default_mask):
        # Create a mask for weights within the range [a, b]
        mask = (tensor >= self.a) & (tensor <= self.b)
        # Invert the mask because we want to prune weights within the range
        return ~mask
    

def range_prune(module, name, alpha, beta):
    # torch.nn.utils.prune.custom_from_mask(module, name=name, mask=RangePruning(alpha, beta).compute_mask(module.weight, default_mask=None))
    RangePruning.apply(module, name, a=alpha, b=beta)
    return module

# custom pruning
class ThresholdPruning(BasePruningMethod):
    PRUNING_TYPE = "unstructured"

    def __init__(self, amount):
        self.s = amount

    def compute_mask(self, tensor, default_mask):
        threshold = torch.std(tensor).item() * self.s
        return torch.abs(tensor) > threshold


# TODO: alternatively prune linear and conv2d
# TODO: iteratively prune and train

def l1_threshold(module, name, amount):
    ThresholdPruning.apply(module, name, amount=amount)
    return module


def get_pruned(module: nn.Module, name: str) -> Tuple[int, int]:
    attr = f"{name}_mask"
    if not hasattr(module, attr):
        return 0, 1
    mask = getattr(module, attr)
    return (mask == 0).sum().item(), mask.numel()


def sparsity_stats(model, name="weight"):
    diff_bits = 5

    sparsity_dict = {n: get_pruned(m, name) for n, m in model.named_modules() if getattr(m, name, None) is not None}
    log.info(f"Sparsity stats of `{model.__class__.__name__}` - `{name}`:")
    for name, (z, p) in sparsity_dict.items():
        log.info(f"  Layer {name}: retained {p - z}/{p} ({(p - z) / p:.2%}) ")
    zeros, params = zip(*sparsity_dict.values())
    total_params = sum(params)
    total_zeros = sum(zeros)
    total_retained = total_params - total_zeros
    log.info(
        "Total:"
        f"  Pruned: {total_zeros}/{total_params} ({total_zeros / total_params:.2%})"
        f"  Retained: {total_retained}/{total_params} ({total_retained / total_params:.2%})"
        f"  Compression: {total_params / total_retained:.1f} X"
    )
    return total_retained, total_zeros