import torch


def compute_hit_rate(topk_items, true_item):
    return int(true_item in topk_items)


def compute_ndcg(topk_items, true_item):
    if true_item in topk_items:
        rank = (topk_items == true_item).nonzero(as_tuple=True)[0].item() + 1
        return 1 / torch.log2(torch.tensor(rank + 1, dtype=torch.float))

    else:
        return 0.0
