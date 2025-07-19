from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn


class LLMSeqRec(nn.Module):
    def __init__(
        self,
        llm_embedding_path: Path,
        num_items: int,
        embedding_dim: int = 64,
        llm_dim: int = 384,
        hidden_size: int = 64,
        max_seq_len: int = 200,
        num_heads: int = 2,
        num_layers: int = 2,
        dropout_rate: float = 0.2,
    ):
        super().__init__()

        self.item_emb = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, embedding_dim)
        self.llm_projection = nn.Linear(llm_dim, hidden_size)

        # Load the pretrained embeddings for the metadata
        llm_np = np.load(llm_embedding_path)  # shape: (num_items, llm_dim)

        # Insert padding vector at index 0
        pad = np.zeros((1, llm_dim), dtype=np.float32)
        llm_np = np.vstack([pad, llm_np])  # shape: (num_items+1, llm_dim)

        # Register as non-trainable buffer
        self.register_buffer("llm_output", torch.tensor(llm_np, dtype=torch.float32))

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

    def forward(self, input_seq: torch.Tensor):
        """
        input_seq: [batch_size, seq_len]
        """
        device = input_seq.device
        seq_len = input_seq.size(1)
        positions = torch.arange(seq_len, device=device).unsqueeze(0)

        # LLM embedding
        llm_raw = self.llm_output[input_seq]  # [B, T, llm_dim]
        llm_embs = self.llm_projection(llm_raw)  # [B, T, D]

        seq_embs = self.item_emb(input_seq) + self.pos_emb(positions) + llm_embs
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
        llm_embs = self.llm_projection(self.llm_output[candidates_tensor])
        item_vecs += llm_embs  # [num_candidates, dim]

        # Compute scores
        scores = torch.matmul(item_vecs, query_vec.squeeze())  # [num_candidates]

        scores = scores.detach().cpu().numpy()
        rank_indices = np.argsort(-scores)
        ranked_items = np.array(candidate_items)[rank_indices]

        return ranked_items[:k].tolist()
