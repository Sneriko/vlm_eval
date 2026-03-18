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
    max_tokens: int = 1024
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
    base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    max_tokens: int = 2048
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
                thinking_config=types.ThinkingConfig(thinking_level="minimal"),
            ),
        )

        return response.text


@dataclass
class DeepSeekClient(OpenAICompatibleClient):
    base_url: str = "https://api.deepseek.com/v1"


def build_client(model_config: ModelConfig):
    api_key = os.environ.get(model_config.api_key_env)
    if not api_key:
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

    raise ValueError(f"Unsupported provider: {model_config.provider}")
