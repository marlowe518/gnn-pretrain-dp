from pathlib import Path
from typing import Tuple

import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid


def load_planetoid(name: str, root: str = "data") -> Tuple[Data, int, int]:
    """
    Load a small citation network dataset (e.g. Cora).
    Returns (data, num_features, num_classes).
    """
    dataset = Planetoid(root=root, name=name)
    data = dataset[0]
    return data, dataset.num_features, dataset.num_classes


def add_or_load_splits(
    data: Data,
    dataset_name: str,
    seed: int,
    cache_dir: Path,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> Data:
    """
    Attach deterministic train/val/test boolean masks to `data`.
    Splits are cached to disk and re-used across runs.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    split_path = cache_dir / f"{dataset_name}_seed{seed}.pt"

    if split_path.exists():
        masks = torch.load(split_path)
        data.train_mask = masks["train_mask"]
        data.val_mask = masks["val_mask"]
        data.test_mask = masks["test_mask"]
        return data

    num_nodes = data.num_nodes
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_nodes, generator=generator)

    n_train = int(train_ratio * num_nodes)
    n_val = int(val_ratio * num_nodes)
    n_test = num_nodes - n_train - n_val

    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val : n_train + n_val + n_test]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    masks = {
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
    }
    torch.save(masks, split_path)

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    return data

