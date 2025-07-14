from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Tuple

from loguru import logger


def preprocess_movie_lens(
    ratings_file: Path, output_file: Path, min_num_actions: int = 5
) -> None:
    user_count, item_count = count_actions(ratings_file)

    users = defaultdict(list)
    with open(ratings_file, "r") as f:
        for line in f:
            user_id, item_id, _, timestamp = line.strip().split("::")
            if (
                user_count[user_id] < min_num_actions
                or item_count[item_id] < min_num_actions
            ):
                continue

            users[user_id].append([timestamp, item_id])

    logger.info(f"Filtered users: {len(users)}")

    item_map = dict()
    item_id_new = 0
    for user_id in users.keys():
        users[user_id].sort(key=lambda x: x[0])

        for item in users[user_id]:
            if item[1] not in item_map:
                item_id_new += 1
                item_map[item[1]] = item_id_new

    logger.info(f"Total unique items after filtering: {len(item_map)}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing preprocessed data to {output_file}")
    f = open(output_file, "w")
    for user_id in users.keys():
        for item in users[user_id]:
            f.write(f"{user_id} {item_map[item[1]]}\n")

    f.close()


def count_actions(
    ratings_file: Path,
) -> Tuple[DefaultDict[int, int], DefaultDict[int, int]]:
    user_count = defaultdict(int)  # Count of interactions per user
    item_count = defaultdict(int)  # Count of interactions per item

    with open(ratings_file, "r") as f:
        for line in f:
            user_id, item_id, _, timestamp = line.strip().split("::")
            user_count[user_id] += 1
            item_count[item_id] += 1

    logger.info(f"User interaction counts: {len(user_count)}")
    logger.info(f"Item interaction counts: {len(item_count)}")

    return user_count, item_count
