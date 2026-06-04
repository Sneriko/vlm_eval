from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass
class ModelConfig:
    name: str
    provider: Literal["openai_compatible", "anthropic", "gemini", "deepseek", "huggingface"]
    model: str
    api_key_env: str | None = None
    base_url: str | None = None
    max_tokens: int = 1024
    temperature: float = 0.0
    device_map: str | None = None
    torch_dtype: str | None = None
    trust_remote_code: bool = False
    model_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalConfig:
    dataset_dir: Path
    image_extensions: list[str] = field(default_factory=lambda: [".jpg", ".jpeg", ".png", ".tif", ".tiff"])
    pagexml_extension: str = ".xml"
    prompt: str = (
        "Transcribe this historical Swedish handwritten page as accurately as possible. "
        "Return only the transcription text."
    )
    output_csv: Path = Path("results.csv")
    models: list[ModelConfig] = field(default_factory=list)


SUPPORTED_PROVIDERS = {"openai_compatible", "anthropic", "gemini", "deepseek", "huggingface"}


def _load_model_config(raw_model: dict) -> ModelConfig:
    provider = raw_model["provider"]
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(f"Unsupported provider in config: {provider}")

    return ModelConfig(
        name=raw_model["name"],
        provider=provider,
        model=raw_model["model"],
        api_key_env=raw_model.get("api_key_env"),
        base_url=raw_model.get("base_url"),
        max_tokens=int(raw_model.get("max_tokens", 1024)),
        temperature=float(raw_model.get("temperature", 0.0)),
        device_map=raw_model.get("device_map"),
        torch_dtype=raw_model.get("torch_dtype"),
        trust_remote_code=bool(raw_model.get("trust_remote_code", False)),
        model_kwargs=dict(raw_model.get("model_kwargs", {})),
    )


def load_config(path: str | Path) -> EvalConfig:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if "dataset_dir" not in data:
        raise ValueError("Missing required field: dataset_dir")

    models = [_load_model_config(model) for model in data.get("models", [])]

    return EvalConfig(
        dataset_dir=Path(data["dataset_dir"]),
        image_extensions=data.get("image_extensions", [".jpg", ".jpeg", ".png", ".tif", ".tiff"]),
        pagexml_extension=data.get("pagexml_extension", ".xml"),
        prompt=data.get(
            "prompt",
            "Transcribe this historical Swedish handwritten page as accurately as possible. Return only the transcription text.",
        ),
        output_csv=Path(data.get("output_csv", "results.csv")),
        models=models,
    )
