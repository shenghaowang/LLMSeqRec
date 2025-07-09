import random
from collections import Counter
from typing import Dict, List

import numpy as np
from loguru import logger


def get_pop_items(u2i: Dict[int, List[int]]) -> List[int]:
    item_counter = Counter()
    for _, items in u2i.items():
        for item in items:
            item_counter[item] += 1

    # Sort items by popularity (most frequent first)
    pop_items = [item for item, _ in item_counter.most_common()]

    return pop_items


def evaluate_poprec(
    train_data: Dict[int, List[int]],
    test_data: Dict[int, List[int]],
    train_pop_items: List[int],
    num_items: int,
    k: int = 10,
    num_negatives: int = 100,
) -> None:
    hit_list = []
    ndcg_list = []

    for user, true_items in test_data.items():
        if not true_items:
            # Skip if no true item is available
            continue

        true_item = true_items[0]
        rated = set(train_data[user])

        # Generate negatives
        negatives = []
        while len(negatives) < num_negatives:
            t = random.randint(1, num_items)
            if t not in rated and t != true_item:
                negatives.append(t)

        candidate_items = [true_item] + negatives

        # Compute popularity of candidates
        pop_scores = [
            train_pop_items.index(item) if item in train_pop_items else float("inf")
            for item in candidate_items
        ]

        # Lower index = more popular, so we sort ascending
        sorted_candidates = [
            item for _, item in sorted(zip(pop_scores, candidate_items))
        ]

        topk = sorted_candidates[:k]

        # Compute hit rate and NDCG
        hit = int(true_item in topk)
        hit_list.append(hit)

        if hit:
            rank = np.where(np.array(topk) == true_item)[0][0] + 1
            ndcg = 1 / np.log2(rank + 1)

        else:
            ndcg = 0.0

        ndcg_list.append(ndcg)

    hit_rate = np.mean(hit_list) if hit_list else 0.0
    ndcg = np.mean(ndcg_list) if ndcg_list else 0.0

    logger.info(f"[PopRec] Hit@{k}: {hit_rate:.4f}, NDCG@{k}: {ndcg:.4f}")
