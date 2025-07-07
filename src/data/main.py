from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from data.bpr_data import BPRDataModule
from data.utils import split_data


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))

    # u2i_index, i2u_index = build_index(Path(cfg.dataset_path))
    # logger.info(f"User to Item Index: {len(u2i_index)} users")
    # logger.info(f"Item to User Index: {len(i2u_index)} items")

    [user_train, user_valid, user_test, num_users, num_items] = split_data(
        Path(cfg.dataset_path)
    )
    logger.info(f"User Train: {len(user_train)} users")
    logger.info(f"User Valid: {len(user_valid)} users")
    logger.info(f"User Test: {len(user_test)} users")
    logger.info(f"Total Users: {num_users}, Total Items: {num_items}")

    dm = BPRDataModule(
        train_interactions=[
            (user, item) for user, items in user_train.items() for item in items
        ],
        val_interactions=[
            (user, item) for user, items in user_valid.items() for item in items
        ],
        test_interactions=[
            (user, item) for user, items in user_test.items() for item in items
        ],
        num_users=num_users,
        num_items=num_items,
        batch_size=2,
    )

    dm.setup()

    # Check dataloader
    train_loader = dm.train_dataloader()
    for batch in train_loader:
        users, pos_items, neg_items = batch
        logger.debug(
            f"Train batch users: {users}",
        )
        logger.debug(f"Train batch pos items: {pos_items}")
        logger.debug(f"Train batch neg items: {neg_items}")

        break


if __name__ == "__main__":
    main()
