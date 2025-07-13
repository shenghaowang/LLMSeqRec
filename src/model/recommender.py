import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from model.matrix_factorization import BPRMatrixFactorization
from model.metrics import compute_hit_rate, compute_ndcg
from model.model_type import ModelType
from model.sasrec import SASRec


class Recommender(pl.LightningModule):
    def __init__(
        self,
        model: BPRMatrixFactorization | SASRec,
        num_items: int,
        lr: float = 0.01,
        k_eval: int = 10,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = model

        # Determine model type
        if isinstance(model, BPRMatrixFactorization):
            self.model_type = ModelType.MatrixFactorization.value

        else:
            self.model_type = ModelType.SASRec.value

        # buffers to collect metrics
        self.val_hits = []
        self.val_ndcgs = []
        self.test_hits = []
        self.test_ndcgs = []

    def forward(self, query, item_ids):
        """
        query:
            - user_ids for MF
            - sequences for SASRec
        item_ids:
            items to score
        """
        if self.model_type == "mf":
            return self.model(query, item_ids)

        else:
            # SASRec: get final hidden state from sequence
            seq_embs = self.model(query)[:, -1, :]  # [batch_size, hidden_dim]
            item_embs = self.model.item_emb(item_ids)
            scores = (seq_embs * item_embs).sum(dim=-1)
            return scores

    def training_step(self, batch, batch_idx):
        query, pos_item, neg_item = batch
        pos_scores = self(query, pos_item)
        neg_scores = self(query, neg_item)

        loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        self.log("train_loss", loss)
        return loss

    def evaluate_batch(self, batch, hits_list, ndcgs_list):
        queries, true_items = batch
        all_items = torch.arange(self.hparams.num_items, device=self.device)

        for query, true_item in zip(queries, true_items):
            if self.model_type == ModelType.MatrixFactorization.value:
                query_vec = self.model.user_emb(query).unsqueeze(0)  # [1, dim]

            else:
                seq = query.unsqueeze(0)
                query_vec = self.model(seq)[:, -1, :]

            item_vecs = self.model.item_emb(all_items)  # [num_items, dim]

            scores = torch.matmul(item_vecs, query_vec.squeeze())  # [num_items]
            topk_items = torch.topk(scores, k=self.hparams.k_eval).indices

            hit = compute_hit_rate(topk_items, true_item)
            hits_list.append(hit)

            ndcg = compute_ndcg(topk_items, true_item)
            ndcgs_list.append(ndcg)

    def validation_step(self, batch, batch_idx):
        self.evaluate_batch(batch, self.val_hits, self.val_ndcgs)

    def on_validation_epoch_end(self):
        hit = (
            torch.tensor(self.val_hits).float().mean().item() if self.val_hits else 0.0
        )
        ndcg = (
            torch.tensor(self.val_ndcgs).float().mean().item()
            if self.val_ndcgs
            else 0.0
        )
        self.log(f"val_hit@{self.hparams.k_eval}", hit)
        self.log(f"val_ndcg@{self.hparams.k_eval}", ndcg)
        self.val_hits.clear()
        self.val_ndcgs.clear()

    def test_step(self, batch, batch_idx):
        self.evaluate_batch(batch, self.test_hits, self.test_ndcgs)

    def on_test_epoch_end(self):
        hit = (
            torch.tensor(self.test_hits).float().mean().item()
            if self.test_hits
            else 0.0
        )
        ndcg = (
            torch.tensor(self.test_ndcgs).float().mean().item()
            if self.test_ndcgs
            else 0.0
        )
        self.log(f"test_hit@{self.hparams.k_eval}", hit)
        self.log(f"test_ndcg@{self.hparams.k_eval}", ndcg)
        self.test_hits.clear()
        self.test_ndcgs.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
