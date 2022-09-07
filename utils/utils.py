"""Misc utils."""
import logging
from pathlib import Path
from typing import List
import argparse

from rich.logging import RichHandler

def parse_args() -> argparse.Namespace:
    """Generate args."""
    parser = argparse.ArgumentParser(description="Simple calculator")
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Which data directory to run.",
        default="data/datasets/entity_matching/structured/iTunes-Amazon"
    )
    parser.add_argument(
        "--output_dir", type=str, help="Output directory.", default="outputs"
    )
    parser.add_argument(
        "--cache_file",
        type=str,
        help="File to cache input/output results.",
        default="openai_cache.sqlite",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite sqlite cache of input/output results.",
    )
    parser.add_argument("--prefix_size", type=int, default=5)
    parser.add_argument('--prefix_or_fine', type=str, default='prefix')
    parser.add_argument("--k", type=int, help="Number examples in prompt", default=1)
    parser.add_argument("--batch_size", type=int, help="Batch size", default=16)
    parser.add_argument("--n_epochs", type=int, help="Amount of epochs", default=10)
    parser.add_argument("--lr", type=float, help="Learning rate", default=0.0001)
    parser.add_argument(
        "--sample_method",
        type=str,
        help="Example generation method",
        default="random",
        choices=["random", "manual", "validation_clusters"],
    )
    parser.add_argument(
        "--validation_path",
        type=str,
        default=None,
        help="Path to validation data error feather file. I.e., the result of \
            running `run_inference` of validation data. Used with \
            validation_clusters method.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--sep_tok",
        type=str,
        help="Separate for attr: val pairs in row. Default is '.'.",
        default=".",
    )
    # Open AI args
    parser.add_argument("--temperature", type=float, help="Temperature.", default=0.0)
    parser.add_argument(
        "--max_tokens", type=int, help="Max tokens to generate.", default=3
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
