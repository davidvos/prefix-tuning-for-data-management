"""Misc utils."""
import argparse

import logging

from pathlib import Path
from typing import List
from rich.logging import RichHandler

def parse_args() -> argparse.Namespace:
    """Generate args."""
    parser = argparse.ArgumentParser(description="Prefix-tuning for DM")
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Which data directory to run.",
        default="data/datasets/entity_matching/structured/iTunes-Amazon"
    )
    parser.add_argument(
        "--output_dir", type=str, help="Output directory.", default="outputs"
    )
    parser.add_argument("--prefix_size", type=int, default=100)
    parser.add_argument('--finetune_type', type=str, default='prefix')
    parser.add_argument('--task', type=str, default='entity-matching')
    parser.add_argument("--n_samples", type=int, help="Ratio of training examples to use", default=0)
    parser.add_argument("--batch_size", type=int, help="Batch size", default=16)
    parser.add_argument("--n_epochs", type=int, help="Amount of epochs", default=20)
    parser.add_argument("--lr", type=float, help="Learning rate", default=5e-5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--sep_tok",
        type=str,
        help="Separate for attr: val pairs in row. Default is '.'.",
        default=".",
    )
    args = parser.parse_args()
    return args

def setup_logger(log_dir: str):
    """Create log directory and logger."""
    Path(log_dir).mkdir(exist_ok=True, parents=True)
    log_path = str(Path(log_dir) / "log.txt")
    handlers = [logging.FileHandler(log_path), RichHandler(rich_tracebacks=True)]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(module)s] [%(levelname)s] %(message)s",
        handlers=handlers,
    )

def compute_metrics(preds: List, golds: List, task: str):
    """Compute metrics."""
    mets = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "crc": 0, "total": 0}

    for pred, label in zip(preds, golds):
        label = label.strip().lower()
        pred = pred.strip().lower()
        mets["total"] += 1
        if task == "data_imputation":
            crc = pred == label
        elif task in {"entity_matching", "error_detection"}:
            crc = pred.startswith(label)
        else:
            raise ValueError(f"Unknown task: {task}")
        # Measure equal accuracy for generation
        if crc:
            mets["crc"] += 1
        if label == "yes":
            if crc:
                mets["tp"] += 1
            else:
                mets["fn"] += 1
        elif label == "no":
            if crc:
                mets["tn"] += 1
            else:
                mets["fp"] += 1

    prec = mets["tp"] / max(1, (mets["tp"] + mets["fp"]))
    rec = mets["tp"] / max(1, (mets["tp"] + mets["fn"]))
    acc = mets["crc"] / mets["total"]
    f1 = 2 * prec * rec / max(1, (prec + rec))
    return prec, rec, acc, f1
