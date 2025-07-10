from typing import List

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.metrics import compute_hit_rate, compute_ndcg


class BPRMatrixFactorization(pl.LightningModule):
    def __init__(self, num_users, num_items, embedding_dim=32, lr=0.01, k_eval=10):
        super().__init__()
        self.save_hyperparameters()

        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        self.lr = lr
        self.k_eval = k_eval

        # buffers to collect metrics
        self.val_hits = []
        self.val_ndcgs = []
        self.test_hits = []
        self.test_ndcgs = []

    def forward(self, user_ids, item_ids):
        user_vecs = self.user_embeddings(user_ids)
        item_vecs = self.item_embeddings(item_ids)
        scores = (user_vecs * item_vecs).sum(dim=1)
        return scores

    def training_step(self, batch, batch_idx):
        user, pos_item, neg_item = batch
        pos_scores = self(user, pos_item)
        neg_scores = self(user, neg_item)

        loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        users, true_items = batch
        all_items = torch.arange(self.hparams.num_items, device=self.device)

        for user, true_item in zip(users, true_items):
            user_vec = self.user_embeddings(user).unsqueeze(0)  # [1, dim]
            item_vecs = self.item_embeddings(all_items)  # [num_items, dim]

            scores = torch.matmul(item_vecs, user_vec.squeeze())  # [num_items]
            topk_items = torch.topk(scores, k=self.k_eval).indices

            hit = compute_hit_rate(topk_items, true_item)
            self.val_hits.append(hit)

            ndcg = compute_ndcg(topk_items, true_item)
            self.val_ndcgs.append(ndcg)

    def on_validation_epoch_end(self):
        recall = (
            torch.tensor(self.val_hits).float().mean().item() if self.val_hits else 0.0
        )
        ndcg = (
            torch.tensor(self.val_ndcgs).float().mean().item()
            if self.val_ndcgs
            else 0.0
        )
        self.log(f"val_recall@{self.k_eval}", recall)
        self.log(f"val_ndcg@{self.k_eval}", ndcg)
        self.val_hits.clear()
        self.val_ndcgs.clear()

    def test_step(self, batch, batch_idx):
        users, true_items = batch
        all_items = torch.arange(self.hparams.num_items, device=self.device)

        for user, true_item in zip(users, true_items):
            user_vec = self.user_embeddings(user).unsqueeze(0)
            item_vecs = self.item_embeddings(all_items)

            scores = torch.matmul(item_vecs, user_vec.squeeze())
            topk_items = torch.topk(scores, k=self.k_eval).indices

            hit = compute_hit_rate(topk_items, true_item)
            self.test_hits.append(hit)

            ndcg = compute_ndcg(topk_items, true_item)
            self.test_ndcgs.append(ndcg)

    def on_test_epoch_end(self):
        recall = (
            torch.tensor(self.test_hits).float().mean().item()
            if self.test_hits
            else 0.0
        )
        ndcg = (
            torch.tensor(self.test_ndcgs).float().mean().item()
            if self.test_ndcgs
            else 0.0
        )
        self.log(f"test_recall@{self.k_eval}", recall)
        self.log(f"test_ndcg@{self.k_eval}", ndcg)
        self.test_hits.clear()
        self.test_ndcgs.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def get_topk_items(self, user: int, candidate_items: List[int]) -> List[int]:
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

        return ranked_items[: self.k_eval]
