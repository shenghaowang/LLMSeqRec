from pathlib import Path

import hydra
from dataset import Dataset
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from preprocess.movie_lens import preprocess_movie_lens


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    logger.info(f"\n{OmegaConf.to_yaml(cfg, resolve=True)}")

    match cfg.dataset:
        case Dataset.MovieLens.value:
            ratings_file = Path(cfg.raw_data.movie_lens.ratings)
            output_file = Path(cfg.dataset_path)
            min_num_actions = cfg.raw_data.min_num_actions

            logger.info(
                f"Preprocessing MovieLens dataset from {ratings_file} to {output_file}"
            )
            preprocess_movie_lens(ratings_file, output_file, min_num_actions)

        case _:
            raise ValueError(f"Dataset {cfg.dataset} is not supported yet.")


if __name__ == "__main__":
    main()
