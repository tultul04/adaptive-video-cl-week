"""
train.py

CLI entry point. Reads a YAML config (see configs/*.yaml) and runs the
continual-learning loop for one method, saving the result to
<output_dir>/result.json.

Usage:
    python train.py --config configs/finetune.yaml
    python train.py --config configs/uniform.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml

from src.cl_loop import run

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    log.info("Loaded config from %s: %s", args.config, config)

    result = run(config)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "result.json"
    out_path.write_text(json.dumps(result, indent=2))
    log.info("Saved result -> %s", out_path)


if __name__ == "__main__":
    main()