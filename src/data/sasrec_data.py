import random

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset


class SASRecDataset(Dataset):
    def __init__(self, user_sequences, num_items, max_seq_len=50):
        """
        user_sequences: dict {user_id: [item1, item2, ...]}
        num_items: total number of items
        """
        self.user_sequences = user_sequences
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.samples = self._prepare_samples()

    def _prepare_samples(self):
        samples = []
        for user, items in self.user_sequences.items():
            for t in range(1, len(items)):
                # items[:t] → history
                # items[t] → next positive
                seq = items[:t]
                pos_item = items[t]

                if len(seq) >= 1:
                    samples.append((user, seq, pos_item))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        user, seq, pos_item = self.samples[idx]

        # Pad sequence
        if len(seq) < self.max_seq_len:
            pad_len = self.max_seq_len - len(seq)
            seq = [0] * pad_len + seq
        else:
            seq = seq[-self.max_seq_len :]  # noqa: E203

        # Negative sampling
        while True:
            neg_item = random.randint(1, self.num_items - 1)
            if neg_item != pos_item and neg_item not in seq:
                break

        return (
            torch.tensor(seq),
            torch.tensor(pos_item),
            torch.tensor(neg_item),
        )


class SASRecEvalDataset(Dataset):
    def __init__(self, user_sequences, max_seq_len=50):
        self.samples = []
        for user, items in user_sequences.items():
            if len(items) >= 2:
                seq = items[:-1]
                true_item = items[-1]
                if len(seq) < max_seq_len:
                    seq = [0] * (max_seq_len - len(seq)) + seq
                else:
                    seq = seq[-max_seq_len:]
                self.samples.append((seq, true_item))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, true_item = self.samples[idx]
        return torch.tensor(seq), torch.tensor(true_item)


class SASRecDataModule(pl.LightningDataModule):
    def __init__(
        self,
        user_sequences,
        num_items,
        batch_size=512,
        max_seq_len=50,
        num_workers=4,
    ):
        super().__init__()
        self.user_sequences = user_sequences
        self.num_items = num_items
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = SASRecDataset(
            user_sequences=self.user_sequences,
            num_items=self.num_items,
            max_seq_len=self.max_seq_len,
        )
        self.val_dataset = SASRecEvalDataset(
            user_sequences=self.user_sequences,
            max_seq_len=self.max_seq_len,
        )
        self.test_dataset = self.val_dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )
