from __future__ import annotations

import json

import click

from .config import load_config
from .evaluator import evaluate, save_csv, summarize


@click.group()
def cli():
    """Evaluate VLM APIs against PAGE XML ground truth."""


@cli.command("run")
@click.option("--config", "config_path", type=click.Path(exists=True, dir_okay=False), required=True)
def run_eval(config_path: str):
    cfg = load_config(config_path)
    rows = evaluate(cfg)
    save_csv(rows, cfg.output_csv)
    click.echo(f"Saved {len(rows)} rows to {cfg.output_csv}")
    click.echo(json.dumps(summarize(rows), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    cli()
