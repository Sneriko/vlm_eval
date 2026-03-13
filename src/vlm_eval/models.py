from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class VisionModelClient(Protocol):
    """Common protocol for VLM API clients."""

    name: str

    def transcribe(self, image_path: str, prompt: str) -> str:
        """Return text transcription from a page image."""


@dataclass
class ModelResult:
    name: str
    prediction: str
    bow_precision: float
    bow_recall: float
    bow_f1: float
