from pathlib import Path

import hydra
from dataset import Dataset
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from sentence_transformers import SentenceTransformer

from preprocess.preprocessor import AmazonReviewsPreprocessor, MovieLensPreprocessor


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    logger.info(f"\n{OmegaConf.to_yaml(cfg, resolve=True)}")
    logger.info(f"Preprocessing dataset: {cfg.dataset}")

    # Initialize pretrained LLM
    model = SentenceTransformer(cfg.preprocess.lm)

    match cfg.dataset:
        case Dataset.MovieLens.value:
            preprocess_cfg = cfg.preprocess.movie_lens
            preprocessor = MovieLensPreprocessor(
                actions_file=Path(preprocess_cfg.actions_file),
                items_file=Path(preprocess_cfg.items_file),
            )

        case Dataset.AmazonGames.value:
            preprocess_cfg = cfg.preprocess.amzn_games
            preprocessor = AmazonReviewsPreprocessor(
                actions_file=Path(preprocess_cfg.actions_file),
                items_file=Path(preprocess_cfg.items_file),
            )

        case _:
            raise ValueError(f"Dataset {cfg.dataset} is not supported yet.")

    user_actions = preprocessor.get_user_actions(
        min_num_actions=cfg.preprocess.min_num_actions
    )
    user_map, item_map = preprocessor.get_user_item_mapping(
        user_actions=user_actions, user_mapping=preprocess_cfg.user_mapping
    )

    output_file = Path(cfg.dataset_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing preprocessed data to {output_file}")
    f = open(output_file, "w")
    for user_id in user_actions.keys():
        for item in user_actions[user_id]:
            user_id_mapped = (
                user_map[user_id] if preprocess_cfg.user_mapping else user_id
            )
            item_id_mapped = item_map[item[1]]
            f.write(f"{user_id_mapped} {item_id_mapped}\n")

    f.close()

    # Load item metadata for embeddings
    item_metadata = preprocessor.load_metadata(
        item_map=item_map,
        prompt=preprocess_cfg.prompt if hasattr(preprocess_cfg, "prompt") else None,
    )

    # Generate embeddings file
    preprocessor.get_item_embeddings(
        item_metadata=item_metadata,
        embedding_file=Path(cfg.preprocess.embedding_path),
        model=model,
        batch_size=cfg.preprocess.batch_size,
    )

    preprocessor.report_data_statistics()


if __name__ == "__main__":
    main()
