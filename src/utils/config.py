import json
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_CONFIG: Dict[str, Any] = {
    "dataset": "Cora",
    "hidden_dim": 64,
    "learning_rate_pretrain": 1e-3,
    "learning_rate_finetune": 1e-2,
    "weight_decay": 5e-4,
    "pretrain_epochs": 50,
    "finetune_epochs": 200,
    "smoke_pretrain_epochs": 2,
    "smoke_finetune_epochs": 5,
}


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load JSON config from disk and merge with defaults.
    """
    config = dict(DEFAULT_CONFIG)
    if config_path is not None:
        path = Path(config_path)
        with path.open("r") as f:
            disk_cfg = json.load(f)
        config.update(disk_cfg)
    return config


def save_config(config: Dict[str, Any], run_dir: Path) -> Path:
    """
    Save the resolved config for this run.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "config.json"
    with path.open("w") as f:
        json.dump(config, f, indent=2, sort_keys=True)
    return path

