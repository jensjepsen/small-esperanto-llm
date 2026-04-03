"""Perplexity evaluation helper."""

import json
import math
from pathlib import Path

from transformers import Trainer


def compute_perplexity(trainer: Trainer) -> float:
    """Compute perplexity on the eval dataset."""
    metrics = trainer.evaluate()
    return math.exp(metrics["eval_loss"])


def save_perplexity(config_name: str, perplexity: float, output_dir: Path = Path("results")):
    """Save perplexity result to a JSON file for later comparison."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "perplexity.json"

    results = {}
    if results_file.exists():
        with open(results_file) as f:
            results = json.load(f)

    results[config_name] = {"perplexity": perplexity}

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
