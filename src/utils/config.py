import json
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_CONFIG: Dict[str, Any] = {
    "dataset": "Cora",
    "hidden_dim": 64,
    "learning_rate_pretrain": 1e-3,
    "learning_rate_finetune": 1e-2,
    # Global finetune LR (used by DP-GNN; others keep their legacy defaults)
    "lr": 3e-3,
    "weight_decay": 5e-4,
    "pretrain_epochs": 50,
    "finetune_epochs": 200,
    "smoke_pretrain_epochs": 2,
    "smoke_finetune_epochs": 5,
    # VAT-related defaults (for DGI+VAT pretraining backend)
    "vat_lambda": 1.0,
    "vat_eps": 1e-2,
    "vat_xi": 1e-6,
    "vat_ip": 1,
    # GAP DP finetuning (when finetune_backend=gap)
    "gap_epsilon": 1.0,
    "gap_delta": "auto",
    "gap_hops": 2,
    "gap_encoder_type": "gnn",
    "gap_debug": False,
    # Node-DP (AP + DP-SGD); only used when gap_privacy == "node"
    "gap_privacy": "edge",
    "gap_max_degree": 200,
    "gap_clip_norm": 1.0,
    "gap_noise_multiplier": 1.0,
    "gap_dp_batch_size": 256,
    "gap_dp_microbatch_size": 1,
    "gap_dp_delta": 1e-5,
    "gap_dp_params": "head_only",
    # DP-GNN finetuning (finetune_backend=dpgnn); upstream: differentially_private_gnns
    "dpgnn_hops": 1,
    "dpgnn_max_degree": 10,
    "dpgnn_pad_to": 0,
    "dpgnn_dp_batch_size": 256,
    "dpgnn_dp_microbatch_size": 1,
    "dpgnn_num_training_steps": 3000,
    "dpgnn_evaluate_every_steps": 50,
    "dpgnn_noise_multiplier": 1.0,
    "dpgnn_delta": "auto",
    "dpgnn_clip_mode": "fixed",
    "dpgnn_clip_norm": 1.0,
    "dpgnn_clip_percentile": 95.0,
    "dpgnn_num_estimation_samples": 500,
    "dpgnn_max_epsilon": 10.0,
    "dpgnn_train_encoder": False,
    "dpgnn_optimizer": "sgd",
    "dpgnn_use_upstream_arch": True,
    "dpgnn_resample_adjacency": False,
}


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load JSON config from disk and merge with defaults.
    If config_path is given, it is resolved to an absolute path so the same
    file is used regardless of later cwd changes.
    """
    config = dict(DEFAULT_CONFIG)
    if config_path is not None:
        path = Path(config_path).resolve()
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

