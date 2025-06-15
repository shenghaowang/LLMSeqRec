from pathlib import Path
from typing import List, Tuple

import numpy as np


def build_index(dataset_path: Path) -> Tuple[List[List[int]], List[List[int]]]:
    """Builds user-to-item and item-to-user indices from a dataset file.

    Parameters
    ----------
    dataset_path : Path
        Path to the dataset file containing user-item interactions.

    Returns
    -------
    Tuple[List[List[int]], List[List[int]]]
        u2i_index: List of lists where each list contains items for a user.
        i2u_index: List of lists where each list contains users for an item.
    """

    ui_mat = np.loadtxt(dataset_path, dtype=np.int32)

    n_users = ui_mat[:, 0].max()
    n_items = ui_mat[:, 1].max()

    u2i_index = [[] for _ in range(n_users + 1)]
    i2u_index = [[] for _ in range(n_items + 1)]

    for u, i in ui_mat:
        u2i_index[u].append(i)
        i2u_index[i].append(u)

    return u2i_index, i2u_index
