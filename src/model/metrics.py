from typing import List

import numpy as np
import torch


def compute_hit_rate(topk_items: torch.Tensor | List[int], true_item: int) -> int:
    if isinstance(topk_items, torch.Tensor):
        return int((topk_items == true_item).any())

    else:
        return int(true_item in topk_items)


def compute_ndcg(topk_items: torch.Tensor | List[int], true_item: int) -> float:
    if isinstance(topk_items, torch.Tensor):
        matches = (topk_items == true_item).nonzero(as_tuple=True)[0]
        if len(matches) == 0:
            return 0.0
        rank = matches.item() + 1

    else:
        indices = np.where(np.array(topk_items) == true_item)[0]
        if len(indices) == 0:
            return 0.0
        rank = indices[0] + 1

    return float(1 / np.log2(rank + 1))
