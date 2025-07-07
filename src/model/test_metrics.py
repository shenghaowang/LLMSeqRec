import pytest
import torch

from model.metrics import compute_hit_rate, compute_ndcg


@pytest.fixture
def topk_items() -> torch.Tensor:
    return torch.tensor([3, 7, 2])


@pytest.fixture
def true_item() -> torch.Tensor:
    return torch.tensor(7)


def test_compute_hit_rate(topk_items: torch.Tensor, true_item: torch.Tensor):
    hit = compute_hit_rate(topk_items, true_item)

    assert hit == 1, f"Expected 1, got {hit}"


def test_compute_ndcg(topk_items: torch.Tensor, true_item: torch.Tensor):
    ndcg = compute_ndcg(topk_items, true_item)

    assert ndcg.item() == pytest.approx(0.6309, abs=1e-4)
