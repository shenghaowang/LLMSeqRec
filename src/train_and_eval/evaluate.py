import random
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from model.matrix_factorization import BPRMatrixFactorization
from model.metrics import compute_hit_rate, compute_ndcg
from model.poprec import PopRec


def evaluate(
    model: PopRec | BPRMatrixFactorization,
    train_data: Dict[int, List[int]],
    test_data: Dict[int, List[int]],
    num_items: int,
    k_eval: int = 10,
    num_negatives: int = 100,
) -> Tuple[float, float]:
    hit_list = []
    ndcg_list = []

    for user, true_items in tqdm(test_data.items()):
        if not true_items:
            # Skip if no true item is available
            continue

        true_item = true_items[0]
        rated = set(train_data[user])

        # Generate negatives
        negatives = []
        while len(negatives) < num_negatives:
            t = random.randint(0, num_items - 1)
            if t not in rated and t != true_item:
                negatives.append(t)

        candidate_items = [true_item] + negatives

        # Get top-k items from the model
        if isinstance(model, PopRec):
            topk = model.get_topk_items(candidate_items)

        else:
            topk = model.get_topk_items(user, candidate_items, k_eval)

        # Compute hit rate and NDCG
        hit = compute_hit_rate(topk, true_item)
        ndcg = compute_ndcg(topk, true_item)

        hit_list.append(hit)
        ndcg_list.append(ndcg)

    hit_rate = np.mean(hit_list) if hit_list else 0.0
    ndcg = np.mean(ndcg_list) if ndcg_list else 0.0

    return hit_rate, ndcg
