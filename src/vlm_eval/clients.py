from __future__ import annotations

import base64
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path

from google import genai
from google.genai import types

from .config import ModelConfig


def _image_to_data_url(image_path: str) -> str:
    image = Path(image_path)
    mime_type = _guess_mime_type(image_path)
    encoded = base64.b64encode(image.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def _guess_mime_type(image_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        raise ValueError(f"Could not determine MIME type for: {image_path}")
    return mime_type


@dataclass
class OpenAICompatibleClient:
    name: str
    model: str
    api_key: str
    base_url: str = "https://api.openai.com/v1"
    max_tokens: int = 1024
    temperature: float = 0.0

    def transcribe(self, image_path: str, prompt: str) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=120.0)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": _image_to_data_url(image_path)},
                        },
                    ],
                }
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        content = response.choices[0].message.content
        return content.strip() if content else ""


@dataclass
class AnthropicClient:
    name: str
    model: str
    api_key: str
    base_url: str = "https://api.anthropic.com"
    max_tokens: int = 4096
    temperature: float = 0.0

    def transcribe(self, image_path: str, prompt: str) -> str:
        from anthropic import Anthropic

        image_bytes = Path(image_path).read_bytes()
        client = Anthropic(api_key=self.api_key, base_url=self.base_url, timeout=120.0)
        response = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": _guess_mime_type(image_path),
                                "data": base64.b64encode(image_bytes).decode("utf-8"),
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        texts = [block.text.strip() for block in response.content if getattr(block, "type", None) == "text" and getattr(block, "text", "").strip()]
        return "\n".join(texts)


@dataclass
class GeminiClient:
    name: str
    model: str
    api_key: str
    max_tokens: int
    base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    temperature: float = 0.0

    def transcribe(self, image_path: str, prompt: str) -> str:
        mime_type = _guess_mime_type(image_path)

        with open(image_path, "rb") as f:
            image_bytes = f.read()

        client = genai.Client(api_key=self.api_key)

        response = client.models.generate_content(
            model=self.model,
            contents=[
                prompt,
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=mime_type,
                ),
            ],
            config=types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                thinking_config=types.ThinkingConfig(thinking_level="low"),
            ),
        )

        return response.text


@dataclass
class HuggingFaceClient:
    name: str
    model: str
    api_key: str | None = None
    max_tokens: int = 1024
    temperature: float = 0.0
    device_map: str | None = "auto"
    torch_dtype: str | None = "auto"
    trust_remote_code: bool = False
    model_kwargs: dict | None = None

    def __post_init__(self):
        self._processor = None
        self._model = None

    def _load(self):
        if self._processor is not None and self._model is not None:
            return

        import transformers
        from transformers import AutoProcessor

        model_kwargs = dict(self.model_kwargs or {})
        if self.api_key:
            model_kwargs.setdefault("token", self.api_key)
        model_kwargs.setdefault("trust_remote_code", self.trust_remote_code)
        if self.device_map:
            model_kwargs.setdefault("device_map", self.device_map)
        if self.torch_dtype:
            model_kwargs.setdefault("torch_dtype", self.torch_dtype)

        processor_kwargs = {"trust_remote_code": self.trust_remote_code}
        if self.api_key:
            processor_kwargs["token"] = self.api_key

        model_cls = getattr(transformers, "AutoModelForImageTextToText", None)
        if model_cls is None:
            model_cls = getattr(transformers, "AutoModelForVision2Seq", None)
        if model_cls is None:
            raise ImportError(
                "Hugging Face VLM support requires a recent transformers release with "
                "AutoModelForImageTextToText or AutoModelForVision2Seq. Install with "
                "`pip install -e .[huggingface]`."
            )

        self._processor = AutoProcessor.from_pretrained(self.model, **processor_kwargs)
        self._model = model_cls.from_pretrained(self.model, **model_kwargs)
        self._model.eval()

    def transcribe(self, image_path: str, prompt: str) -> str:
        import torch
        from PIL import Image

        self._load()
        assert self._processor is not None
        assert self._model is not None

        image = Image.open(image_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "path": str(Path(image_path))},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        try:
            inputs = self._processor.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
            )
        except (TypeError, ValueError):
            text = self._processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self._processor(text=[text], images=[image], return_tensors="pt")

        device = getattr(self._model, "device", None)
        if device is not None and str(device) != "meta" and hasattr(inputs, "to"):
            inputs = inputs.to(device)

        generate_kwargs = {
            "max_new_tokens": self.max_tokens,
            "do_sample": self.temperature > 0,
        }
        if self.temperature > 0:
            generate_kwargs["temperature"] = self.temperature

        with torch.inference_mode():
            output_ids = self._model.generate(**inputs, **generate_kwargs)

        input_length = inputs["input_ids"].shape[-1]
        generated_ids = output_ids[:, input_length:]
        decoded = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return decoded[0].strip() if decoded else ""


@dataclass
class DeepSeekClient(OpenAICompatibleClient):
    base_url: str = "https://api.deepseek.com/v1"


def build_client(model_config: ModelConfig):
    api_key = os.environ.get(model_config.api_key_env) if model_config.api_key_env else None
    if model_config.provider != "huggingface" and not api_key:
        raise ValueError(
            f"Missing API key for model '{model_config.name}'. "
            f"Set environment variable {model_config.api_key_env}."
        )

    if model_config.provider == "openai_compatible":
        return OpenAICompatibleClient(
            name=model_config.name,
            model=model_config.model,
            api_key=api_key,
            base_url=model_config.base_url or "https://api.openai.com/v1",
            max_tokens=model_config.max_tokens,
            temperature=model_config.temperature,
        )

    if model_config.provider == "anthropic":
        return AnthropicClient(
            name=model_config.name,
            model=model_config.model,
            api_key=api_key,
            base_url=model_config.base_url or "https://api.anthropic.com",
            max_tokens=model_config.max_tokens,
            temperature=model_config.temperature,
        )

    if model_config.provider == "gemini":
        return GeminiClient(
            name=model_config.name,
            model=model_config.model,
            api_key=api_key,
            base_url=model_config.base_url or "https://generativelanguage.googleapis.com/v1beta",
            max_tokens=model_config.max_tokens,
            temperature=model_config.temperature,
        )

    if model_config.provider == "deepseek":
        return DeepSeekClient(
            name=model_config.name,
            model=model_config.model,
            api_key=api_key,
            base_url=model_config.base_url or "https://api.deepseek.com/v1",
            max_tokens=model_config.max_tokens,
            temperature=model_config.temperature,
        )

    if model_config.provider == "huggingface":
        return HuggingFaceClient(
            name=model_config.name,
            model=model_config.model,
            api_key=api_key,
            max_tokens=model_config.max_tokens,
            temperature=model_config.temperature,
            device_map=model_config.device_map or "auto",
            torch_dtype=model_config.torch_dtype or "auto",
            trust_remote_code=model_config.trust_remote_code,
            model_kwargs=model_config.model_kwargs,
        )

    raise ValueError(f"Unsupported provider: {model_config.provider}")
