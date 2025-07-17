import gzip
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Tuple

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer


class Preprocessor(ABC):
    def __init__(self, actions_file: Path, items_file: Path):
        super().__init__()

        self.actions_file = actions_file
        self.items_file = items_file
        self.user_count, self.item_count = self.count_actions()

    def count_actions(self) -> Tuple[DefaultDict[int, int], DefaultDict[int, int]]:
        raise NotImplementedError("The count_action method is not implemented.")

    @abstractmethod
    def get_user_actions(
        self, min_num_actions: int
    ) -> DefaultDict[int, List[Tuple[int, float]]]:
        raise NotImplementedError("The get_user_actions method is not implemented.")

    @staticmethod
    def get_user_item_mapping(
        user_actions: DefaultDict[int, List[Tuple[int, float]]],
        user_mapping: bool = False,
    ) -> Dict[str, int]:
        if user_mapping:
            user_map = dict()
            user_id_new = 0

        else:
            user_map = None

        item_map = dict()
        item_id_new = 0
        for user_id in user_actions.keys():
            if user_mapping:
                if user_id not in user_map:
                    user_id_new += 1
                    user_map[user_id] = user_id_new

            user_actions[user_id].sort(key=lambda x: x[0])

            for item in user_actions[user_id]:
                if item[1] not in item_map:
                    item_id_new += 1
                    item_map[item[1]] = item_id_new

        logger.info(f"Total unique items after filtering: {len(item_map)}")

        return user_map, item_map

    @abstractmethod
    def load_metadata(self, item_map: Dict[str, int], prompt: str = None) -> List[str]:
        raise NotImplementedError("The load_metadata method is not implemented.")

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
    def __init__(self, actions_file: Path, items_file: Path):
        super().__init__(actions_file=actions_file, items_file=items_file)

    def count_actions(self) -> Tuple[DefaultDict[int, int], DefaultDict[int, int]]:
        user_count = defaultdict(int)  # Count of interactions per user
        item_count = defaultdict(int)  # Count of interactions per item

        with open(self.actions_file, "r") as f:
            for line in f:
                user_id, item_id, _, _ = line.strip().split("::")
                user_count[user_id] += 1
                item_count[item_id] += 1

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

    def load_metadata(self, item_map: Dict[str, int], prompt: str) -> List[str]:
        # Load metadata for embeddings
        item_metadata = [None] * len(item_map)
        with open(self.items_file, "r", encoding="latin1") as f:
            for line in f:
                item_id, title, genres = line.strip().split("::")
                if item_id in item_map:
                    item_id_new = item_map[item_id]
                    item_metadata[item_id_new - 1] = prompt.format(
                        title=title, genres=",".join(genres.strip().split("|"))
                    )

        return item_metadata


class AmazonReviewsPreprocessor(Preprocessor):
    def __init__(self, actions_file: Path, items_file: Path):
        super().__init__(actions_file=actions_file, items_file=items_file)

        self.items_missing_description = self._get_items_missing_description(
            items_file=items_file
        )

    @staticmethod
    def _parse(file_path: Path):
        file = gzip.open(file_path, "r")
        for line in file:
            yield eval(line)

    def count_actions(self) -> Tuple[DefaultDict[int, int], DefaultDict[int, int]]:
        user_count = defaultdict(int)  # Count of interactions per user
        item_count = defaultdict(int)  # Count of interactions per item

        for review in self._parse(self.actions_file):
            user_count[review["reviewerID"]] += 1
            item_count[review["asin"]] += 1

        return user_count, item_count

    def get_user_actions(
        self, min_num_actions: int = 5
    ) -> DefaultDict[str, List[Tuple[str, str]]]:
        user_actions = defaultdict(list)
        for review in self._parse(self.actions_file):
            user_id = review["reviewerID"]
            item_id = review["asin"]
            timestamp = review["unixReviewTime"]

            if (
                self.user_count[user_id] < min_num_actions
                or self.item_count[item_id] < min_num_actions
                or item_id in self.items_missing_description
            ):
                continue

            user_actions[user_id].append((timestamp, item_id))

        logger.info(f"Filtered users: {len(user_actions)}")

        return user_actions

    def load_metadata(self, item_map: Dict[str, int], prompt: str = None) -> List[str]:
        # Load metadata for embeddings
        item_metadata = [None] * len(item_map)
        for item in self._parse(self.items_file):
            if item["asin"] in item_map:
                item_id_new = item_map[item["asin"]]
                item_metadata[item_id_new - 1] = item["description"]

        return item_metadata

    def _get_items_missing_description(self, items_file: Path) -> List[str]:
        items_missing_description = []

        for item in self._parse(items_file):
            if "description" not in item:
                items_missing_description.append(item["asin"])

        logger.info(
            f"There are {len(items_missing_description)} items without description."
        )

        return items_missing_description
