import random
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from model.matrix_factorization import BPRMatrixFactorization
from model.metrics import compute_hit_rate, compute_ndcg
from model.poprec import PopRec
from model.sasrec import SASRec


def evaluate(
    model: PopRec | BPRMatrixFactorization | SASRec,
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

        elif isinstance(model, SASRec):
            # Get user sequence from training data
            user_seq = train_data[user]
            if len(user_seq) == 0:
                continue

            # Truncate or pad user_seq to model's max_seq_len
            max_seq_len = getattr(model, "max_seq_len", 200)
            if len(user_seq) < max_seq_len:
                pad_len = max_seq_len - len(user_seq)
                user_seq = [0] * pad_len + user_seq

            else:
                user_seq = user_seq[-max_seq_len:]

            # Get top-k items from the model
            topk = model.get_topk_items(user_seq, candidate_items, k_eval)

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
