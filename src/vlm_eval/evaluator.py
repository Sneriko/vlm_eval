from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from .clients import build_client
from .config import EvalConfig
from .metrics import bow_scores
from .pagexml import parse_pagexml_text


@dataclass
class EvalRow:
    level: str
    scope_id: str
    image_path: str
    xml_path: str
    model: str
    bow_precision: float
    bow_recall: float
    bow_f1: float
    prediction: str
    reference: str


def find_samples(dataset_dir: Path, image_extensions: list[str], pagexml_extension: str):
    xml_files = sorted(dataset_dir.rglob(f"*{pagexml_extension}"))
    for xml_path in xml_files:
        if xml_path.parent.name != "page":
            continue
        stem = xml_path.stem
        image_path = None
        archive_dir = xml_path.parent.parent
        for ext in image_extensions:
            candidate = archive_dir / f"{stem}{ext}"
            if candidate.exists():
                image_path = candidate
                break
        if image_path is None:
            continue
        yield image_path, xml_path


def evaluate(config: EvalConfig, progress_logger: Callable[[str], None] | None = None) -> list[EvalRow]:
    rows: list[EvalRow] = []

    clients = [build_client(model_cfg) for model_cfg in config.models]
    samples_by_folder: dict[Path, list[tuple[Path, Path]]] = {}
    for image_path, xml_path in find_samples(config.dataset_dir, config.image_extensions, config.pagexml_extension):
        folder_path = xml_path.parent.parent
        samples_by_folder.setdefault(folder_path, []).append((image_path, xml_path))

    testset_predictions: dict[str, list[str]] = {client.name: [] for client in clients}
    testset_references: dict[str, list[str]] = {client.name: [] for client in clients}

    for folder_path in sorted(samples_by_folder):
        folder_id = str(folder_path.relative_to(config.dataset_dir))
        folder_predictions: dict[str, list[str]] = {client.name: [] for client in clients}
        folder_references: dict[str, list[str]] = {client.name: [] for client in clients}

        for image_path, xml_path in sorted(samples_by_folder[folder_path], key=lambda sample: sample[1]):
            reference = parse_pagexml_text(xml_path)
            page_id = str(xml_path.relative_to(config.dataset_dir))
            for client in clients:
                prediction = client.transcribe(str(image_path), config.prompt)
                scores = bow_scores(reference, prediction)
                rows.append(
                    EvalRow(
                        level="page",
                        scope_id=page_id,
                        image_path=str(image_path),
                        xml_path=str(xml_path),
                        model=client.name,
                        bow_precision=scores.precision,
                        bow_recall=scores.recall,
                        bow_f1=scores.f1,
                        prediction=prediction,
                        reference=reference,
                    )
                )
                folder_predictions[client.name].append(prediction)
                folder_references[client.name].append(reference)
                testset_predictions[client.name].append(prediction)
                testset_references[client.name].append(reference)

                if progress_logger:
                    progress_logger(
                        f"[PAGE] model={client.name} page={page_id} "
                        f"precision={scores.precision:.4f} recall={scores.recall:.4f} f1={scores.f1:.4f}"
                    )

        for client in clients:
            folder_reference = "\n".join(folder_references[client.name])
            folder_prediction = "\n".join(folder_predictions[client.name])
            scores = bow_scores(folder_reference, folder_prediction)
            rows.append(
                EvalRow(
                    level="folder",
                    scope_id=folder_id,
                    image_path="",
                    xml_path="",
                    model=client.name,
                    bow_precision=scores.precision,
                    bow_recall=scores.recall,
                    bow_f1=scores.f1,
                    prediction="",
                    reference="",
                )
            )
            if progress_logger:
                progress_logger(
                    f"[FOLDER] model={client.name} folder={folder_id} "
                    f"precision={scores.precision:.4f} recall={scores.recall:.4f} f1={scores.f1:.4f}"
                )

    for client in clients:
        testset_reference = "\n".join(testset_references[client.name])
        testset_prediction = "\n".join(testset_predictions[client.name])
        scores = bow_scores(testset_reference, testset_prediction)
        rows.append(
            EvalRow(
                level="testset",
                scope_id="entire_testset",
                image_path="",
                xml_path="",
                model=client.name,
                bow_precision=scores.precision,
                bow_recall=scores.recall,
                bow_f1=scores.f1,
                prediction="",
                reference="",
            )
        )

    return rows


def save_csv(rows: list[EvalRow], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()) if rows else [
            "level", "scope_id", "image_path", "xml_path", "model", "bow_precision", "bow_recall", "bow_f1", "prediction", "reference"
        ])
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def summarize(rows: list[EvalRow]) -> dict[str, dict[str, float]]:
    summary_rows = [row for row in rows if row.level == "testset"] or rows

    by_model: dict[str, list[EvalRow]] = {}
    for row in summary_rows:
        by_model.setdefault(row.model, []).append(row)

    summary: dict[str, dict[str, float]] = {}
    for model, model_rows in by_model.items():
        n = len(model_rows)
        summary[model] = {
            "bow_precision": sum(r.bow_precision for r in model_rows) / n,
            "bow_recall": sum(r.bow_recall for r in model_rows) / n,
            "bow_f1": sum(r.bow_f1 for r in model_rows) / n,
            "samples": float(n),
        }

    return summary
