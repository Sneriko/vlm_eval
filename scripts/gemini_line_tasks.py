#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from google import genai
from google.genai import types


@dataclass
class CerAccumulator:
    total_edits: int = 0
    total_chars: int = 0

    def add(self, edits: int, reference_len: int) -> None:
        self.total_edits += edits
        self.total_chars += reference_len

    @property
    def cer(self) -> float:
        if self.total_chars == 0:
            return 0.0
        return self.total_edits / self.total_chars


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            insert_cost = curr[j - 1] + 1
            delete_cost = prev[j] + 1
            replace_cost = prev[j - 1] + (ca != cb)
            curr.append(min(insert_cost, delete_cost, replace_cost))
        prev = curr
    return prev[-1]


def cer(reference: str, hypothesis: str) -> tuple[float, int, int]:
    edits = levenshtein_distance(reference, hypothesis)
    ref_len = len(reference)
    if ref_len == 0:
        return (0.0 if not hypothesis else 1.0), edits, ref_len
    return edits / ref_len, edits, ref_len


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run 6 Gemini transcription/correction tasks for line_predictions.csv "
            "and compute per-line + dataset CER."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("data/line_predictions.csv"),
        help="Input CSV with cropped_image_path, trocr_transcription, ground_truth_transcription.",
    )
    parser.add_argument(
        "--guidelines-pdf",
        type=Path,
        default=Path("data/tridis.pdf"),
        help="PDF with transcription guidelines used for tasks 4, 5, and 6.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/line_predictions_gemini_tasks.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--model",
        default="gemini-3.1-pro-preview",
        help="Gemini model name.",
    )
    parser.add_argument(
        "--api-key-env",
        default="GEMINI_API_KEY",
        help="Environment variable name holding Gemini API key.",
    )
    parser.add_argument(
        "--image-path-prefix",
        type=Path,
        default=None,
        help=(
            "Optional local prefix prepended to image basename if cropped_image_path in CSV "
            "does not exist on this machine."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for quick trial runs.",
    )
    return parser


def resolve_image_path(raw_path: str, image_path_prefix: Path | None) -> Path:
    candidate = Path(raw_path)
    if candidate.exists():
        return candidate
    if image_path_prefix is not None:
        prefixed = image_path_prefix / candidate.name
        if prefixed.exists():
            return prefixed
    raise FileNotFoundError(f"Image not found: {raw_path}")


def _pdf_part(guidelines_pdf: Path) -> types.Part:
    return types.Part.from_bytes(
        data=guidelines_pdf.read_bytes(),
        mime_type="application/pdf",
    )


def _image_part(image_path: Path) -> types.Part:
    suffix = image_path.suffix.lower()
    mime_type = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
        ".webp": "image/webp",
    }.get(suffix)
    if mime_type is None:
        raise ValueError(f"Unsupported image format for Gemini tasking: {image_path}")
    return types.Part.from_bytes(data=image_path.read_bytes(), mime_type=mime_type)


def generate_text(client: genai.Client, model: str, parts: Iterable[types.Part | str]) -> str:
    response = client.models.generate_content(
        model=model,
        contents=list(parts),
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=512,
            thinking_config=types.ThinkingConfig(thinking_level="low"),
        ),
    )
    return (response.text or "").strip()


def task_prompts(trocr_text: str) -> dict[str, str]:
    return {
        "task1": (
            "Transcribe the text in the provided line image. "
            "Return only the transcription and nothing else."
        ),
        "task2": (
            "You will receive a handwritten line image and an initial OCR transcription. "
            "Correct the OCR text based on the image. "
            "Return only the corrected transcription.\n"
            f"Initial OCR transcription: {trocr_text}"
        ),
        "task3": (
            "Correct the following OCR transcription without seeing the image. "
            "Return only the corrected transcription.\n"
            f"OCR transcription: {trocr_text}"
        ),
        "task4": (
            "Correct the following OCR transcription without seeing the image. "
            "Follow the attached transcription guidelines PDF. "
            "Return only the corrected transcription.\n"
            f"OCR transcription: {trocr_text}"
        ),
        "task5": (
            "Correct the provided OCR transcription using both the handwritten image "
            "and the attached transcription guidelines PDF. "
            "Return only the corrected transcription.\n"
            f"OCR transcription: {trocr_text}"
        ),
        "task6": (
            "Transcribe the handwritten line image using the attached transcription "
            "guidelines PDF. Return only the transcription."
        ),
    }


def main() -> None:
    args = build_parser().parse_args()
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise ValueError(f"Missing API key in environment variable: {args.api_key_env}")

    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")
    if not args.guidelines_pdf.exists():
        raise FileNotFoundError(f"Guidelines PDF not found: {args.guidelines_pdf}")

    client = genai.Client(api_key=api_key)
    pdf_guidelines = _pdf_part(args.guidelines_pdf)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    with args.input_csv.open("r", newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        rows = list(reader)

    if args.limit is not None:
        rows = rows[: args.limit]

    task_accumulators = {f"task{i}": CerAccumulator() for i in range(1, 7)}
    output_rows: list[dict[str, str]] = []

    for idx, row in enumerate(rows, start=1):
        image_path = resolve_image_path(row["cropped_image_path"], args.image_path_prefix)
        image_part = _image_part(image_path)
        gt = row["ground_truth_transcription"]
        trocr = row["trocr_transcription"]
        prompts = task_prompts(trocr)

        task_outputs = {
            "task1": generate_text(client, args.model, [prompts["task1"], image_part]),
            "task2": generate_text(client, args.model, [prompts["task2"], image_part]),
            "task3": generate_text(client, args.model, [prompts["task3"]]),
            "task4": generate_text(client, args.model, [prompts["task4"], pdf_guidelines]),
            "task5": generate_text(client, args.model, [prompts["task5"], image_part, pdf_guidelines]),
            "task6": generate_text(client, args.model, [prompts["task6"], image_part, pdf_guidelines]),
        }

        out_row = dict(row)
        for task_name, transcription in task_outputs.items():
            task_cer, edits, ref_len = cer(gt, transcription)
            out_row[f"gemini_{task_name}_transcription"] = transcription
            out_row[f"gemini_{task_name}_cer"] = f"{task_cer:.6f}"
            task_accumulators[task_name].add(edits, ref_len)

        output_rows.append(out_row)
        print(f"Processed {idx}/{len(rows)}: {row.get('line_key', str(idx))}")

    fieldnames = list(output_rows[0].keys()) if output_rows else []
    summary_row = {key: "" for key in fieldnames}
    summary_row["line_key"] = "__SUMMARY__"
    for task_name, acc in task_accumulators.items():
        summary_key = f"gemini_{task_name}_dataset_cer"
        summary_row[summary_key] = f"{acc.cer:.6f}"
        if summary_key not in fieldnames:
            fieldnames.append(summary_key)

    with args.output_csv.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
        writer.writerow(summary_row)

    print("Dataset CER summary:")
    for task_name, acc in task_accumulators.items():
        print(f"  {task_name}: {acc.cer:.6f}")
    print(f"Wrote: {args.output_csv}")


if __name__ == "__main__":
    main()
