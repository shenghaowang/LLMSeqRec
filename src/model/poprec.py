from collections import Counter
from typing import Dict, List


class PopRec:
    def __init__(
        self, train_u2i: Dict[int, List[int]], num_items: int, k_eval: int = 10
    ):
        self.k_eval = k_eval
        self.num_items = num_items
        self.sorted_items = self._sort_items_by_freq(train_u2i)

    def _sort_items_by_freq(self, train_u2i: Dict[int, List[int]]) -> List[int]:
        item_counter = Counter()
        for _, items in train_u2i.items():
            for item in items:
                item_counter[item] += 1

        # Sort items by popularity (most frequent first)
        sorted_items = [item for item, _ in item_counter.most_common()]

        return sorted_items

    def get_topk_items(self, candidate_items: List[int]) -> List[int]:
        # Compute popularity of candidates
        pop_scores = [
            self.sorted_items.index(item) if item in self.sorted_items else float("inf")
            for item in candidate_items
        ]

        # Lower index = more popular, so we sort ascending
        sorted_candidates = [
            item for _, item in sorted(zip(pop_scores, candidate_items))
        ]

        return sorted_candidates[: self.k_eval]
