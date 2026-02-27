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
from src.utils.config import load_config, save_config


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
    pretrainer: DGIPretrainer,
    finetune_trainer: NodeClassificationTrainer,
    logger,
) -> None:
    logger("Running dry run (1 DGI step + 1 classifier step).")
    pre_info = pretrainer.dry_run_debug_step()
    fin_info = finetune_trainer.dry_run_debug_step()
    logger(f"[dry] DGI loss={pre_info['pretrain_loss']:.4f}, grad_norm={pre_info['grad_norm']:.4f}")
    logger(f"[dry] CLS loss={fin_info['train_loss']:.4f}, grad_norm={fin_info['grad_norm']:.4f}")
    logger("Dry run completed successfully.")


def run_training_and_eval(
    mode: str,
    pretrainer: DGIPretrainer,
    finetune_trainer: NodeClassificationTrainer,
    pretrain_epochs: int,
    finetune_epochs: int,
    logger,
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}

    if mode in ("pretrain", "full"):
        logger(f"Starting DGI pretraining for {pretrain_epochs} epochs.")
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

    args = parser.parse_args()

    if args.dry_run and args.smoke_test:
        raise ValueError("Use only one of --dry_run or --smoke_test.")

    set_seeds(args.seed)

    run_dir = create_run_dir(args.mode, args.dry_run, args.smoke_test, args.run_id)
    log_path = run_dir / "log.txt"
    logger = create_logger(log_path)

    logger(f"Run directory: {run_dir}")
    logger(f"Mode={args.mode}, dry_run={args.dry_run}, smoke_test={args.smoke_test}")

    # Load config and resolve epoch counts based on mode.
    config = load_config(args.config)
    config["seed"] = args.seed
    config_path = save_config(config, run_dir)
    logger(f"Config saved to {config_path}")

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

    pretrainer = DGIPretrainer(
        model=dgi_model,
        data=data,
        lr=float(config.get("learning_rate_pretrain", 1e-3)),
        weight_decay=float(config.get("weight_decay", 5e-4)),
        device=device,
        logger=logger,
    )

    finetune_trainer = NodeClassificationTrainer(
        encoder=encoder,
        num_classes=num_classes,
        data=data,
        lr=float(config.get("learning_rate_finetune", 1e-2)),
        weight_decay=float(config.get("weight_decay", 5e-4)),
        device=device,
        logger=logger,
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
    )

    metrics_path = run_dir / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    logger(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()

