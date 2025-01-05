import torch
import torch.nn as nn
import math
import warnings


class SpecAug(nn.Module):
    def __init__(self, num_freq_masks, freq_mask_range, freq_mask_ratio_range, num_time_masks, time_mask_range, time_mask_ratio_range):
        super().__init__()
        self.num_freq_masks = num_freq_masks
        self.freq_mask_range = freq_mask_range
        self.freq_mask_ratio_range = freq_mask_ratio_range
        self.num_time_masks = num_time_masks
        self.time_mask_range = time_mask_range
        self.time_mask_ratio_range = time_mask_ratio_range


    def forward(self, x): # (batch, freq, time)
        x = apply_masking(x, self.num_freq_masks, self.freq_mask_range, self.freq_mask_ratio_range, axis=1)
        x = apply_masking(x, self.num_time_masks, self.time_mask_range, self.time_mask_ratio_range, axis=2)
        return x
    


def apply_masking(x, num_masks, mask_range, mask_ratio_range, axis):
    if mask_range is None:
        min_mask_range = max(0, math.floor(mask_ratio_range[0] * x.size(axis)))
        max_mask_range = min(x.size(axis), math.floor(mask_ratio_range[1] * x.size(axis)))
        mask_range = (min_mask_range, max_mask_range)
    
    if mask_range[0] > mask_range[1]:
        warnings.warn("mask_range[0] is greater than mask_range[1]. Swapping.")
        return x
    
    return mask_along_axis(x, num_masks, mask_range, axis)



def mask_along_axis(x, num_masks, mask_range, axis):
    batch = x.size(0)
    D = x.size(axis)

    mask_len = torch.randint(mask_range[0], mask_range[1], (batch, num_masks), device=x.device).unsqueeze(2)
    mask_pos = torch.randint(0, max(1, D - mask_len.max()), (batch, num_masks), device=x.device).unsqueeze(2)

    arange = torch.arange(D, device=x.device).view(1, 1, -1)

    mask = (mask_pos <= arange) & (arange < mask_pos + mask_len)
    mask = mask.any(dim = 1) # Multiply masks along the num_masks dimension: (batch, num_masks, D) -> (batch, D)

    if axis == 1:
        mask = mask.unsqueeze(2)
    elif axis == 2:
        mask = mask.unsqueeze(1)
    else:
        raise ValueError("Only 1 and 2 are valid axis values")
    
    return x.masked_fill(mask, 0.0)


