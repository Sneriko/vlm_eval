from __future__ import annotations

import json

import click

from .config import load_config
from .evaluator import evaluate, save_csv_per_model, summarize


@click.group()
def cli():
    """Evaluate VLM APIs against PAGE XML ground truth."""


@cli.command("run")
@click.option("--config", "config_path", type=click.Path(exists=True, dir_okay=False), required=True)
def run_eval(config_path: str):
    cfg = load_config(config_path)
    rows = evaluate(cfg, progress_logger=click.echo)
    output_paths = save_csv_per_model(rows, cfg.output_csv)
    for model_name, output_path in output_paths.items():
        row_count = sum(1 for row in rows if row.model == model_name)
        click.echo(f"Saved {row_count} rows for {model_name} to {output_path}")
    click.echo(json.dumps(summarize(rows), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    cli()
