from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Tuple

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer


class Preprocessor(ABC):
    def __init__(
        self,
        actions_file: Path,
        # metadata_file: Path,
        # output_file: Path,
        # embedding_file: Path,
        # model: SentenceTransformer,
        # prompt: str,
        # min_num_actions: int = 5,
        # batch_size: int = 128,
    ):
        super().__init__()

        self.actions_file = actions_file
        self.user_count, self.item_count = self.count_actions()

    def count_actions(self) -> Tuple[DefaultDict[int, int], DefaultDict[int, int]]:
        raise NotImplementedError("The count_action method is not implemented.")

    @abstractmethod
    def get_user_actions(
        self, min_num_actions: int
    ) -> DefaultDict[int, List[Tuple[int, float]]]:
        raise NotImplementedError("The get_user_actions method is not implemented.")

    @staticmethod
    def get_item_mapping(
        user_actions: DefaultDict[int, List[Tuple[int, float]]]
    ) -> Dict[str, int]:
        item_map = dict()
        item_id_new = 0
        for user_id in user_actions.keys():
            user_actions[user_id].sort(key=lambda x: x[0])

            for item in user_actions[user_id]:
                if item[1] not in item_map:
                    item_id_new += 1
                    item_map[item[1]] = item_id_new

        logger.info(f"Total unique items after filtering: {len(item_map)}")

        return item_map

    @staticmethod
    def get_item_embeddings(
        item_metadata: List[str],
        embedding_file: Path,
        model: SentenceTransformer,
        batch_size: int = 128,
    ) -> None:

        # Generate embeddings from metadata
        embeddings = model.encode(
            item_metadata,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=True,
        )
        logger.debug(f"Embeddings shape: {embeddings.shape}")

        np.save(embedding_file, embeddings)
        logger.info(f"Embeddings saved to {embedding_file}")

    def report_data_statistics(self) -> None:
        # Report the number of unique users and items
        logger.info(f"Number of unique users: {len(self.user_count)}")
        logger.info(f"Number of unique items: {len(self.item_count)}")

        # Report the average number of actions per user and
        # the average number of actions per item
        logger.info(
            f"Average actions per user: {np.mean(list(self.user_count.values())):.2f}"
        )
        logger.info(
            f"Average actions per item: {np.mean(list(self.item_count.values())):.2f}"
        )

        # Report the total number of actions
        logger.info(f"Total number of actions: {sum(self.user_count.values())}")


class MovieLensPreprocessor(Preprocessor):
    def __init__(self, actions_file: Path):
        super().__init__(actions_file=actions_file)

    def count_actions(self) -> Tuple[DefaultDict[int, int], DefaultDict[int, int]]:
        user_count = defaultdict(int)  # Count of interactions per user
        item_count = defaultdict(int)  # Count of interactions per item

        with open(self.actions_file, "r") as f:
            for line in f:
                user_id, item_id, _, _ = line.strip().split("::")
                user_count[user_id] += 1
                item_count[item_id] += 1

        logger.info(f"User interaction counts: {len(user_count)}")
        logger.info(f"Item interaction counts: {len(item_count)}")

        return user_count, item_count

    def get_user_actions(
        self,
        min_num_actions: int = 5,
    ) -> DefaultDict[str, List[Tuple[str, str]]]:

        user_actions = defaultdict(list)
        with open(self.actions_file, "r") as f:
            for line in f:
                user_id, item_id, _, timestamp = line.strip().split("::")
                if (
                    self.user_count[user_id] < min_num_actions
                    or self.item_count[item_id] < min_num_actions
                ):
                    continue

                user_actions[user_id].append((timestamp, item_id))

        logger.info(f"Filtered users: {len(user_actions)}")

        return user_actions

    def load_metadata(
        self, item_map: Dict[str, int], items_file: Path, prompt: str
    ) -> List[str]:
        # Load metadata for embeddings
        item_metadata = [None] * len(item_map)
        with open(items_file, "r", encoding="latin1") as f:
            for line in f:
                item_id, title, genres = line.strip().split("::")
                if item_id in item_map:
                    item_id_new = item_map[item_id]
                    item_metadata[item_id_new - 1] = prompt.format(
                        title=title, genres=",".join(genres.strip().split("|"))
                    )

        return item_metadata
