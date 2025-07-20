from typing import List

import numpy as np
import torch
import torch.nn as nn


class SASRec(nn.Module):
    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 64,
        hidden_size: int = 64,
        max_seq_len: int = 200,
        num_heads: int = 2,
        num_layers: int = 2,
        dropout_rate: float = 0.2,
    ):
        super().__init__()

        self.max_seq_len = max_seq_len
        self.item_emb = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_size,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

    def forward(self, input_seq):
        """
        input_seq: [batch_size, seq_len]
        """
        positions = torch.arange(input_seq.size(1), device=input_seq.device).unsqueeze(
            0
        )

        seq_embs = self.item_emb(input_seq) + self.pos_emb(positions)
        mask = input_seq == 0

        output = self.transformer_encoder(
            seq_embs,
            src_key_padding_mask=mask,
        )
        return output

    def get_topk_items(
        self,
        user_sequence: List[int],
        candidate_items: List[int],
        k: int,
    ) -> List[int]:
        """
        Get the top-k items for a given set of candidate items,
        for a sequential model like SASRec.

        :param user_sequence: List of item indices in user's history.
        :param candidate_items: List of candidate item indices.
        :param k: Top-k to return.
        :return: List of top-k item indices.
        """
        # Convert to tensors
        seq = torch.tensor(user_sequence).unsqueeze(0)  # [1, seq_len]
        candidates_tensor = torch.tensor(candidate_items)

        # Compute user (sequence) embedding
        query_vec = self(seq)[:, -1, :]  # [1, dim]

        # Get candidate item vectors
        item_vecs = self.item_emb(candidates_tensor)  # [num_candidates, dim]

        # Compute scores
        scores = torch.matmul(item_vecs, query_vec.squeeze())  # [num_candidates]

        scores = scores.detach().cpu().numpy()
        rank_indices = np.argsort(-scores)
        ranked_items = np.array(candidate_items)[rank_indices]

        return ranked_items[:k].tolist()
