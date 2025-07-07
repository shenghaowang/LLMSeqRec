import random
from typing import List, Tuple

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset


# --- BPRDataset for Training ---
class BPRDataset(Dataset):
    def __init__(self, user_item_pairs: List[Tuple[int, int]], num_items: int):
        """_summary_

        Parameters
        ----------
        user_item_pairs : List[Tuple[int, int]]
            list of (user, item) interactions
        num_items : int
            total number of items
        """
        self.user_item_pairs = user_item_pairs  # list of (u, i)
        self.num_items = num_items
        self.user_pos_dict = self._build_user_pos_dict()

    def _build_user_pos_dict(self):
        d = {}
        for u, i in self.user_item_pairs:
            d.setdefault(u, set()).add(i)
        return d

    def __len__(self):
        return len(self.user_item_pairs)

    def __getitem__(self, idx):
        user, pos_item = self.user_item_pairs[idx]
        # Negative sampling
        while True:
            neg_item = random.randint(0, self.num_items - 1)
            if neg_item not in self.user_pos_dict[user]:
                break
        return torch.tensor(user), torch.tensor(pos_item), torch.tensor(neg_item)


# --- Val/Test Dataset ---
class UserItemDataset(Dataset):
    def __init__(self, user_item_pairs):
        """
        Args:
            user_item_pairs: list of (user, ground_truth_item)
        """
        self.user_item_pairs = user_item_pairs

    def __len__(self):
        return len(self.user_item_pairs)

    def __getitem__(self, idx):
        user, true_item = self.user_item_pairs[idx]
        return torch.tensor(user), torch.tensor(true_item)


class BPRDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_interactions,
        val_interactions=None,
        test_interactions=None,
        num_users=None,
        num_items=None,
        batch_size=1024,
        num_workers=4,
    ):
        """
        Args:
            train_interactions: list of (user, item) pairs for training
            val_interactions: list of (user, item) pairs for validation
            test_interactions: list of (user, item) pairs for test
            num_users: total number of users
            num_items: total number of items
            batch_size: DataLoader batch size
        """
        super().__init__()
        self.train_interactions = train_interactions
        self.val_interactions = val_interactions
        self.test_interactions = test_interactions
        self.num_users = num_users
        self.num_items = num_items
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = BPRDataset(
            user_item_pairs=self.train_interactions,
            num_items=self.num_items,
        )

        self.val_dataset = UserItemDataset(self.val_interactions)
        self.test_dataset = UserItemDataset(self.test_interactions)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
