from pathlib import Path

import hydra
from dataset import Dataset
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from sentence_transformers import SentenceTransformer

# from preprocess.movie_lens import preprocess_movie_lens
# from preprocess.amzn_reviews import process_amzn_reviews
from preprocess.preprocessor import MovieLensPreprocessor


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    logger.info(f"\n{OmegaConf.to_yaml(cfg, resolve=True)}")
    logger.info(f"Preprocessing dataset: {cfg.dataset}")

    # Initialize pretrained LLM
    model = SentenceTransformer(cfg.preprocess.lm)

    match cfg.dataset:
        case Dataset.MovieLens.value:
            preprocessor = MovieLensPreprocessor(
                actions_file=Path(cfg.preprocess.movie_lens.ratings),
            )

        # case Dataset.AmazonGames.value:
        #     process_amzn_reviews(
        #         reviews_file=Path(cfg.preprocess.amzn_games.reviews),
        #         output_file=Path(cfg.dataset_path),
        #     )

        case _:
            raise ValueError(f"Dataset {cfg.dataset} is not supported yet.")

    user_actions = preprocessor.get_user_actions(
        min_num_actions=cfg.preprocess.min_num_actions
    )
    item_map = preprocessor.get_item_mapping(user_actions=user_actions)

    output_file = Path(cfg.dataset_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing preprocessed data to {output_file}")
    f = open(output_file, "w")
    for user_id in user_actions.keys():
        for item in user_actions[user_id]:
            f.write(f"{user_id} {item_map[item[1]]}\n")

    f.close()

    # Load item metadata for embeddings
    item_metadata = preprocessor.load_metadata(
        item_map=item_map,
        items_file=Path(cfg.preprocess.movie_lens.movies),
        prompt=cfg.preprocess.movie_lens.prompt,
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
