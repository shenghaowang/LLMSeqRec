from pathlib import Path

import hydra
from dataset import Dataset
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from sentence_transformers import SentenceTransformer

from preprocess.movie_lens import preprocess_movie_lens


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    logger.info(f"\n{OmegaConf.to_yaml(cfg, resolve=True)}")
    logger.info(f"Preprocessing dataset: {cfg.dataset}")

    # Initialize pretrained LLM
    model = SentenceTransformer("all-MiniLM-L6-v2")

    match cfg.dataset:
        case Dataset.MovieLens.value:
            preprocess_movie_lens(
                ratings_file=Path(cfg.preprocess.movie_lens.ratings),
                movies_file=Path(cfg.preprocess.movie_lens.movies),
                output_file=Path(cfg.dataset_path),
                embedding_file=Path(cfg.preprocess.embedding_path),
                model=model,
                prompt=cfg.preprocess.movie_lens.prompt,
                min_num_actions=cfg.preprocess.min_num_actions,
                batch_size=cfg.preprocess.batch_size,
            )

        case _:
            raise ValueError(f"Dataset {cfg.dataset} is not supported yet.")


if __name__ == "__main__":
    main()
