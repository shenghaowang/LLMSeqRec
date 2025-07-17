from dataclasses import dataclass
from enum import Enum


class Dataset(Enum):
    MovieLens = "ml-1m"
    AmazonGames = "amzn_games"


@dataclass
class AmazonReviewAttributes:
    user_id: str = "reviewerID"
    item_id: str = "asin"
    timestamp: str = "unixReviewTime"
    description: str = "description"
