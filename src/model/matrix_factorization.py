from typing import List

import numpy as np
import torch
import torch.nn as nn


class BPRMatrixFactorization(torch.nn.Module):
    def __init__(self, num_users: int, num_items: int, embedding_dim=16):
        super().__init__()

        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)

    def forward(self, user_ids, item_ids):
        user_vecs = self.user_emb(user_ids)
        item_vecs = self.item_emb(item_ids)
        scores = (user_vecs * item_vecs).sum(dim=1)

        return scores

    def get_topk_items(
        self, user: int, candidate_items: List[int], k: int
    ) -> List[int]:
        """
        Get the top-k items for a given set of candidate items.
        :param candidate_items: List of candidate item indices.
        :return: List of top-k item indices.
        """
        # print(f"Max item_id: {max(candidate_items)}.")

        candidates_tensor = torch.tensor(candidate_items)
        user_tensor = torch.tensor([user] * len(candidate_items))

        # Compute scores
        scores = self(user_tensor, candidates_tensor)
        scores = scores.detach().cpu().numpy()

        rank_indices = np.argsort(-scores)
        ranked_items = np.array(candidate_items)[rank_indices]

        return ranked_items[:k]
