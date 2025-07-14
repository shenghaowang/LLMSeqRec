from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Tuple

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer


def preprocess_movie_lens(
    ratings_file: Path,
    movies_file: Path,
    output_file: Path,
    embedding_file: Path,
    model: SentenceTransformer,
    prompt: str,
    min_num_actions: int = 5,
    batch_size: int = 128,
) -> None:

    # Count user and item interactions
    user_count, item_count = count_actions(ratings_file)

    # Get user actions with filtering based on min_num_actions
    users = get_user_actions(
        ratings_file=ratings_file,
        user_count=user_count,
        item_count=item_count,
        min_num_actions=min_num_actions,
    )

    # Remap item IDs to continuous integers starting from 1
    item_map = get_item_mapping(users)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing preprocessed data to {output_file}")
    f = open(output_file, "w")
    for user_id in users.keys():
        for item in users[user_id]:
            f.write(f"{user_id} {item_map[item[1]]}\n")

    f.close()

    # Load metadata for embeddings
    item_metadata = [None] * len(item_map)
    with open(movies_file, "r", encoding="latin1") as f:
        for line in f:
            item_id, title, genres = line.strip().split("::")
            if item_id in item_map:
                item_id_new = item_map[item_id]
                item_metadata[item_id_new - 1] = prompt.format(
                    title=title, genres=",".join(genres.strip().split("|"))
                )

    # Generate embeddings file
    embeddings = model.encode(
        item_metadata,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=True,
    )
    logger.debug(f"Embeddings shape: {embeddings.shape}")

    np.save(embedding_file, embeddings)
    logger.info(f"Embeddings saved to {embedding_file}")


def count_actions(
    ratings_file: Path,
) -> Tuple[DefaultDict[int, int], DefaultDict[int, int]]:
    user_count = defaultdict(int)  # Count of interactions per user
    item_count = defaultdict(int)  # Count of interactions per item

    with open(ratings_file, "r") as f:
        for line in f:
            user_id, item_id, _, _ = line.strip().split("::")
            user_count[user_id] += 1
            item_count[item_id] += 1

    logger.info(f"User interaction counts: {len(user_count)}")
    logger.info(f"Item interaction counts: {len(item_count)}")

    return user_count, item_count


def get_user_actions(
    ratings_file: Path,
    user_count: DefaultDict[str, int],
    item_count: DefaultDict[str, int],
    min_num_actions: int = 5,
) -> DefaultDict[str, List[Tuple[str, str]]]:

    user_actions = defaultdict(list)
    with open(ratings_file, "r") as f:
        for line in f:
            user_id, item_id, _, timestamp = line.strip().split("::")
            if (
                user_count[user_id] < min_num_actions
                or item_count[item_id] < min_num_actions
            ):
                continue

            user_actions[user_id].append((timestamp, item_id))

    logger.info(f"Filtered users: {len(user_actions)}")

    return user_actions


def get_item_mapping(users: DefaultDict[str, list]) -> Dict[str, int]:
    item_map = dict()
    item_id_new = 0
    for user_id in users.keys():
        users[user_id].sort(key=lambda x: x[0])

        for item in users[user_id]:
            if item[1] not in item_map:
                item_id_new += 1
                item_map[item[1]] = item_id_new

    logger.info(f"Total unique items after filtering: {len(item_map)}")

    return item_map
