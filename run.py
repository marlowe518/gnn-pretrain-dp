import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from src.data.datasets import add_or_load_splits, load_planetoid
from src.finetune.node_classifier_trainer import NodeClassificationTrainer
from src.models.dgi import build_dgi_model
from src.pretrain.dgi_trainer import DGIPretrainer
from src.pretrain.dgi_vat_trainer import DGIVATPretrainer
from src.utils.config import load_config, save_config

try:
    from src.dp.gap_trainer import GAPFinetuneTrainer
except Exception:
    GAPFinetuneTrainer = None


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_run_dir(mode: str, dry_run: bool, smoke_test: bool, run_id: str | None) -> Path:
    if run_id is None:
        suffix = []
        if dry_run:
            suffix.append("dry")
        if smoke_test:
            suffix.append("smoke")
        suffix_str = ("_" + "-".join(suffix)) if suffix else ""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        run_id = f"{timestamp}_{mode}{suffix_str}"
    run_dir = Path("outputs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def create_logger(log_path: Path):
    log_file = log_path.open("a")

    def log(msg: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        line = f"[{timestamp}] {msg}"
        print(line)
        log_file.write(line + "\n")
        log_file.flush()

    return log


def run_dry_run(
    pretrainer,
    finetune_trainer,
    logger,
) -> None:
    logger("Running dry run (1 pretrain step + 1 classifier step).")
    pre_info = pretrainer.dry_run_debug_step()
    fin_info = finetune_trainer.dry_run_debug_step()
    logger(f"[dry] DGI loss={pre_info['pretrain_loss']:.4f}, grad_norm={pre_info['grad_norm']:.4f}")
    logger(f"[dry] CLS loss={fin_info['train_loss']:.4f}, grad_norm={fin_info['grad_norm']:.4f}")
    logger("Dry run completed successfully.")


def run_training_and_eval(
    mode: str,
    pretrainer,
    finetune_trainer,
    pretrain_epochs: int,
    finetune_epochs: int,
    logger,
    pretrain_backend: str,
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}

    if mode in ("pretrain", "full"):
        logger(
            f"Starting pretraining backend={pretrain_backend} "
            f"for {pretrain_epochs} epochs."
        )
        pre_metrics = pretrainer.train(pretrain_epochs)
        metrics["pretrain"] = pre_metrics

    if mode in ("finetune", "full"):
        logger(f"Starting node classification finetuning for {finetune_epochs} epochs.")
        fin_metrics = finetune_trainer.train(finetune_epochs)
        eval_results = {}
        for split in ("train", "val", "test"):
            eval_results[split] = finetune_trainer.evaluate(split)
            logger(
                f"[eval] split={split} "
                f"accuracy={eval_results[split]['accuracy']:.3f}"
            )
        metrics["finetune"] = {
            "train": eval_results["train"],
            "val": eval_results["val"],
            "test": eval_results["test"],
            **fin_metrics,
        }

    return metrics


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["pretrain", "finetune", "full"],
        default="full",
        help="Which part of the pipeline to run.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Run a 1-step debug pass.")
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Run a short training run (1-3 epochs) for verification.",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config.")
    parser.add_argument("--run_id", type=str, default=None, help="Optional run id.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--pretrain_backend",
        choices=["dgi", "dgi_vat"],
        default="dgi",
        help="Pretraining backend: dgi (Deep Graph Infomax) or dgi_vat (DGI + VAT).",
    )
    parser.add_argument(
        "--finetune_backend",
        choices=["vanilla", "gap"],
        default="vanilla",
        help="Finetuning backend: vanilla (standard) or gap (DP via Graph Aggregation Perturbation).",
    )
    parser.add_argument(
        "--gap_debug",
        action="store_true",
        help="Enable extra GAP DP debug logging and checks.",
    )
    parser.add_argument(
        "--gap_debug_strict",
        action="store_true",
        help="Treat GAP debug checks as hard assertions (may raise errors).",
    )
    parser.add_argument(
        "--gap_debug_resample_check",
        action="store_true",
        help="Run additional GAP resampling sanity check (debug-only, cheap).",
    )
    parser.add_argument(
        "--finetune_train_encoder",
        action="store_true",
        help="Finetune encoder + head (full finetune). Mutually exclusive with --finetune_freeze_encoder.",
    )
    parser.add_argument(
        "--finetune_freeze_encoder",
        action="store_true",
        help="Freeze encoder, train head only (linear probe). Mutually exclusive with --finetune_train_encoder.",
    )
    parser.add_argument(
        "--gap_encoder_type",
        choices=["gnn", "mlp"],
        default=None,
        help="GAP finetune encoder type: gnn (DGI encoder) or mlp (feature-only). Default from config or gnn.",
    )
    parser.add_argument(
        "--gap_privacy",
        choices=["edge", "node"],
        default=None,
        help="GAP privacy: edge (AP only) or node (AP + DP-SGD). Default from config or edge.",
    )
    parser.add_argument("--gap_max_degree", type=int, default=None, help="Max degree for node-DP bounded sampling.")
    parser.add_argument("--gap_clip_norm", type=float, default=None, help="DP-SGD gradient clip norm (L2).")
    parser.add_argument("--gap_noise_multiplier", type=float, default=None, help="DP-SGD noise multiplier (sigma).")
    parser.add_argument("--gap_dp_batch_size", type=int, default=None, help="DP-SGD batch size (lot size).")
    parser.add_argument("--gap_dp_microbatch_size", type=int, default=None, help="DP-SGD microbatch size.")
    parser.add_argument("--gap_dp_delta", type=float, default=None, help="DP-SGD delta (for (eps,delta)-DP).")
    parser.add_argument(
        "--gap_dp_params",
        action="store_true",
        default=None,
        help="Include encoder in DP-SGD (node-DP). Default from config.",
    )
    parser.add_argument("--no_gap_dp_params", action="store_true", help="Exclude encoder from DP-SGD (head only).")

    args = parser.parse_args()

    if args.dry_run and args.smoke_test:
        raise ValueError("Use only one of --dry_run or --smoke_test.")
    if args.finetune_train_encoder and args.finetune_freeze_encoder:
        raise ValueError("Cannot set both --finetune_train_encoder and --finetune_freeze_encoder.")
    if getattr(args, "gap_dp_params", False) and getattr(args, "no_gap_dp_params", False):
        raise ValueError("Cannot set both --gap_dp_params and --no_gap_dp_params.")

    set_seeds(args.seed)

    run_dir = create_run_dir(args.mode, args.dry_run, args.smoke_test, args.run_id)
    log_path = run_dir / "log.txt"
    logger = create_logger(log_path)

    logger(f"Run directory: {run_dir}")
    logger(
        f"Mode={args.mode}, dry_run={args.dry_run}, smoke_test={args.smoke_test}, "
        f"pretrain_backend={args.pretrain_backend}, "
        f"finetune_backend={args.finetune_backend}"
    )

    # Load config and resolve epoch counts based on mode.
    if args.config is not None:
        from pathlib import Path as _Path

        config_path_abs = _Path(args.config).resolve()
        logger(f"Loaded config file: {config_path_abs}")
    else:
        config_path_abs = None

    config = load_config(args.config)
    config["seed"] = args.seed
    config_path = save_config(config, run_dir)
    logger(f"Config saved to {config_path}")

    # Effective GAP debug flags (finetune_backend may still be non-GAP).
    gap_debug = bool(args.gap_debug or config.get("gap_debug", False))
    gap_debug_strict = bool(args.gap_debug_strict)
    gap_debug_resample_check = bool(args.gap_debug_resample_check)

    # Resolve train_encoder: CLI overrides config; else backend default.
    cfg_train = config.get("finetune_train_encoder")
    cfg_freeze = config.get("finetune_freeze_encoder")
    if cfg_train is not None and cfg_freeze is not None and cfg_train and cfg_freeze:
        raise ValueError("Config cannot set both finetune_train_encoder and finetune_freeze_encoder to true.")
    if args.finetune_train_encoder:
        train_encoder = True
    elif args.finetune_freeze_encoder:
        train_encoder = False
    elif cfg_train is not None and cfg_train:
        train_encoder = True
    elif cfg_freeze is not None and cfg_freeze:
        train_encoder = False
    else:
        train_encoder = True if args.finetune_backend == "vanilla" else False
    logger(f"finetune_backend={args.finetune_backend}, train_encoder={train_encoder}")

    # GAP encoder type: CLI > config > default "gnn"
    gap_encoder_type = (
        args.gap_encoder_type
        if args.gap_encoder_type is not None
        else str(config.get("gap_encoder_type", "gnn")).lower()
    )
    if gap_encoder_type not in ("gnn", "mlp"):
        gap_encoder_type = "gnn"

    if args.smoke_test:
        pretrain_epochs = int(config.get("smoke_pretrain_epochs", 2))
        finetune_epochs = int(config.get("smoke_finetune_epochs", 5))
    else:
        pretrain_epochs = int(config.get("pretrain_epochs", 50))
        finetune_epochs = int(config.get("finetune_epochs", 200))

    # Data
    dataset_name = config.get("dataset", "Cora")
    data, in_channels, num_classes = load_planetoid(dataset_name)
    splits_cache_dir = Path("outputs") / "splits"
    data = add_or_load_splits(
        data=data,
        dataset_name=dataset_name,
        seed=args.seed,
        cache_dir=splits_cache_dir,
    )
    logger(
        f"Loaded dataset {dataset_name} with "
        f"{data.num_nodes} nodes, {data.num_edges} edges."
    )

    # Models
    hidden_dim = int(config.get("hidden_dim", 64))
    dgi_model, encoder = build_dgi_model(in_channels, hidden_dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger(f"Using device: {device}")

    if args.pretrain_backend == "dgi_vat":
        pretrainer = DGIVATPretrainer(
            model=dgi_model,
            encoder=encoder,
            data=data,
            num_classes=num_classes,
            lr=float(config.get("learning_rate_pretrain", 1e-3)),
            weight_decay=float(config.get("weight_decay", 5e-4)),
            device=device,
            logger=logger,
            vat_lambda=float(config.get("vat_lambda", 1.0)),
            vat_eps=float(config.get("vat_eps", 1e-2)),
            vat_xi=float(config.get("vat_xi", 1e-6)),
            vat_ip=int(config.get("vat_ip", 1)),
        )
    else:
        pretrainer = DGIPretrainer(
            model=dgi_model,
            data=data,
            lr=float(config.get("learning_rate_pretrain", 1e-3)),
            weight_decay=float(config.get("weight_decay", 5e-4)),
            device=device,
            logger=logger,
        )

    if args.finetune_backend == "gap":
        if GAPFinetuneTrainer is None:
            raise RuntimeError("GAP backend requested but src.dp.gap_trainer could not be loaded (check external/GAP and dependencies).")
        gap_epsilon = float(config.get("gap_epsilon", 1.0))
        gap_delta = config.get("gap_delta", "auto")
        gap_hops = int(config.get("gap_hops", 2))
        if gap_debug:
            logger(
                "[GAP-DEBUG] "
                f"epsilon={gap_epsilon}, delta={gap_delta}, hops={gap_hops}, "
                f"config_path={config_path_abs}"
            )
        # Extension point: pretrained encoder to use for GAP (gnn = DGI encoder; mlp = None for now).
        gnn_encoder_ckpt = encoder  # DGI/GNN encoder (used when gap_encoder_type=="gnn")
        mlp_encoder_ckpt = None  # Future: when pretrain_method in {"mlpinit","supervised_mlp"}, load MLP ckpt here
        if gap_encoder_type == "gnn":
            encoder_for_gap = gnn_encoder_ckpt
            load_pretrained_encoder = True
        elif gap_encoder_type == "mlp":
            from src.models.mlp_encoder import MLPEncoder
            encoder_for_gap = MLPEncoder(
                in_channels=in_channels,
                hidden_dim=hidden_dim,
                out_dim=hidden_dim,
                num_layers=2,
                dropout=0.5,
                use_bn=False,
            )
            # Future: if pretrain_method in {"mlpinit", "supervised_mlp"} and mlp_encoder_ckpt: load state_dict
            load_pretrained_encoder = mlp_encoder_ckpt is not None
        else:
            raise ValueError(f"gap_encoder_type must be 'gnn' or 'mlp', got {gap_encoder_type!r}")
        # Node-DP (AP + DP-SGD) params: CLI overrides config; defaults for edge-only
        gap_privacy = (
            args.gap_privacy if args.gap_privacy is not None else str(config.get("gap_privacy", "edge")).lower()
        )
        if gap_privacy not in ("edge", "node"):
            gap_privacy = "edge"
        gap_max_degree = int(config.get("gap_max_degree", 10)) if args.gap_max_degree is None else args.gap_max_degree
        gap_clip_norm = float(config.get("gap_clip_norm", 1.0)) if args.gap_clip_norm is None else args.gap_clip_norm
        gap_noise_multiplier = (
            float(config.get("gap_noise_multiplier", 1.0))
            if args.gap_noise_multiplier is None
            else args.gap_noise_multiplier
        )
        gap_dp_batch_size = (
            int(config.get("gap_dp_batch_size", 256)) if args.gap_dp_batch_size is None else args.gap_dp_batch_size
        )
        gap_dp_microbatch_size = (
            int(config.get("gap_dp_microbatch_size", 64))
            if args.gap_dp_microbatch_size is None
            else args.gap_dp_microbatch_size
        )
        gap_dp_delta = (
            float(config.get("gap_dp_delta", 1e-5)) if args.gap_dp_delta is None else args.gap_dp_delta
        )
        if args.gap_dp_params:
            gap_dp_params = True
        elif args.no_gap_dp_params:
            gap_dp_params = False
        else:
            gap_dp_params = bool(config.get("gap_dp_params", True))
        if gap_privacy == "node":
            logger(
                f"[GAP-DP] privacy=node, max_degree={gap_max_degree}, clip_norm={gap_clip_norm}, "
                f"noise_mult={gap_noise_multiplier}, batch={gap_dp_batch_size}, microbatch={gap_dp_microbatch_size}, "
                f"delta={gap_dp_delta}, dp_params={gap_dp_params}"
            )
        logger(
            f"[FINETUNE] backend=gap, gap_encoder_type={gap_encoder_type}, "
            f"load_pretrained_encoder={load_pretrained_encoder}"
        )
        finetune_trainer = GAPFinetuneTrainer(
            encoder=encoder_for_gap,
            num_classes=num_classes,
            data=data,
            lr=float(config.get("learning_rate_finetune", 1e-2)),
            weight_decay=float(config.get("weight_decay", 5e-4)),
            device=device,
            logger=logger,
            epsilon=gap_epsilon,
            delta=gap_delta,
            hops=gap_hops,
            gap_debug=gap_debug,
            gap_debug_strict=gap_debug_strict,
            gap_debug_resample_check=gap_debug_resample_check,
            train_encoder=train_encoder,
            gap_encoder_type=gap_encoder_type,
            gap_privacy=gap_privacy,
            gap_max_degree=gap_max_degree,
            gap_clip_norm=gap_clip_norm,
            gap_noise_multiplier=gap_noise_multiplier,
            gap_dp_batch_size=gap_dp_batch_size,
            gap_dp_microbatch_size=gap_dp_microbatch_size,
            gap_dp_delta=gap_dp_delta,
            gap_dp_params=gap_dp_params,
        )
    else:
        finetune_trainer = NodeClassificationTrainer(
            encoder=encoder,
            num_classes=num_classes,
            data=data,
            lr=float(config.get("learning_rate_finetune", 1e-2)),
            weight_decay=float(config.get("weight_decay", 5e-4)),
            device=device,
            logger=logger,
            train_encoder=train_encoder,
        )

    if args.dry_run:
        run_dry_run(pretrainer, finetune_trainer, logger)
        return

    metrics = run_training_and_eval(
        mode=args.mode,
        pretrainer=pretrainer,
        finetune_trainer=finetune_trainer,
        pretrain_epochs=pretrain_epochs,
        finetune_epochs=finetune_epochs,
        logger=logger,
        pretrain_backend=args.pretrain_backend,
    )

    metrics_path = run_dir / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    logger(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()

