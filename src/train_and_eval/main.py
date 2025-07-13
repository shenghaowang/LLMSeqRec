from pathlib import Path

import hydra
import pytorch_lightning as pl
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from data.bpr_data import BPRDataModule
from data.sasrec_data import SASRecDataModule
from data.utils import split_data
from model.matrix_factorization import BPRMatrixFactorization
from model.model_type import ModelType
from model.poprec import PopRec
from model.recommender import Recommender
from model.sasrec import SASRec
from train_and_eval.evaluate import evaluate


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    logger.info(f"\n{OmegaConf.to_yaml(cfg, resolve=True)}")
    pl.seed_everything(42)

    [user_train, user_valid, user_test, num_users, num_items] = split_data(
        Path(cfg.dataset_path)
    )
    logger.info(f"Total Users: {num_users}, Total Items: {num_items}")

    recommender = None

    if cfg.model.name == ModelType.PopRec.value:
        model = PopRec(train_u2i=user_train, num_items=num_items)

    elif cfg.model.name == ModelType.MatrixFactorization.value:
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
            batch_size=cfg.train.batch_size,
        )
        dm.setup()

        model = BPRMatrixFactorization(
            num_users=num_users,
            num_items=num_items,
            **cfg.model.hparams,
        )
        recommender = Recommender(
            model=model,
            num_items=num_items,
            lr=cfg.train.lr,
            k_eval=cfg.train.k_eval,
        )

        trainer = pl.Trainer(
            accelerator=cfg.train.device,
            max_epochs=5,
            logger=False,
            enable_checkpointing=False,
        )
        trainer.fit(recommender, dm)

    elif cfg.model.name == ModelType.SASRec.value:
        dm = SASRecDataModule(
            user_sequences=user_train,
            num_items=num_items,
            batch_size=cfg.train.batch_size,
            max_seq_len=cfg.model.hparams.max_seq_len,
        )
        dm.setup()

        model = SASRec(
            num_items=num_items,
            **cfg.model.hparams,
        )
        recommender = Recommender(
            model=model,
            num_items=num_items,
            lr=cfg.train.lr,
            k_eval=cfg.train.k_eval,
        )

        trainer = pl.Trainer(
            accelerator=cfg.train.device,
            max_epochs=5,
            logger=False,
            enable_checkpointing=False,
        )
        trainer.fit(recommender, dm)

    else:
        raise ValueError(f"Unsupported model type: {cfg.model.name}")

    hit_rate, ndcg = evaluate(
        model=recommender.model if recommender is not None else model,
        train_data=user_train,
        test_data=user_test,
        num_items=num_items,
        k_eval=cfg.train.k_eval,
    )

    logger.info(
        f"[{cfg.model.name}] Hit@{cfg.train.k_eval}: {hit_rate:.4f}, NDCG@{cfg.train.k_eval}: {ndcg:.4f}"
    )


if __name__ == "__main__":
    main()
