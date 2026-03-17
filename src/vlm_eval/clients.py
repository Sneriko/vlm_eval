from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from google import genai
from google.genai import types
from .config import ModelConfig
import mimetypes


def _image_to_data_url(image_path: str) -> str:
    image = Path(image_path)
    suffix = image.suffix.lower().replace(".", "") or "png"
    mime = "jpeg" if suffix == "jpg" else suffix
    encoded = base64.b64encode(image.read_bytes()).decode("utf-8")
    return f"data:image/{mime};base64,{encoded}"


def _image_media_type(image_path: str) -> str:
    media_type = f"image/{Path(image_path).suffix.lower().replace('.', '') or 'png'}"
    if media_type == "image/jpg":
        return "image/jpeg"
    if media_type == "image/tif":
        return "image/tiff"
    return media_type


@dataclass
class OpenAICompatibleClient:
    name: str
    model: str
    api_key: str
    base_url: str = "https://api.openai.com/v1"
    max_tokens: int = 1024
    temperature: float = 0.0

    def transcribe(self, image_path: str, prompt: str) -> str:
        import requests

        response = requests.post(
            f"{self.base_url.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [
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
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            },
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()
        return payload["choices"][0]["message"]["content"].strip()


@dataclass
class AnthropicClient:
    name: str
    model: str
    api_key: str
    base_url: str = "https://api.anthropic.com/v1"
    max_tokens: int = 1024
    temperature: float = 0.0

    def transcribe(self, image_path: str, prompt: str) -> str:
        import requests

        image_bytes = Path(image_path).read_bytes()

        response = requests.post(
            f"{self.base_url.rstrip('/')}/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": _image_media_type(image_path),
                                    "data": base64.b64encode(image_bytes).decode("utf-8"),
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            },
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()
        text_blocks = [block.get("text", "") for block in payload.get("content", []) if block.get("type") == "text"]
        return "\n".join(block.strip() for block in text_blocks if block.strip())


@dataclass
class GeminiClient:
    name: str
    model: str
    api_key: str
    base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    max_tokens: int = 2048
    temperature: float = 0.0

    def transcribe(self, image_path: str, prompt: str) -> str:
        print(image_path)

        mime_type, _ = mimetypes.guess_type(image_path)
        if mime_type is None:
            raise ValueError(f"Could not determine MIME type for: {image_path}")

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
            ),
        )

        print(response.text + "\n")
        return response.text
        
        """
        import requests

        response = requests.post(
            f"{self.base_url.rstrip('/')}/models/{self.model}:generateContent",
            params={"key": self.api_key},
            headers={"Content-Type": "application/json"},
            json={
                "contents": [
                    {
                        "parts": [
                            {"text": prompt},
                            {
                                "inline_data": {
                                    "mime_type": _image_media_type(image_path),
                                    "data": base64.b64encode(Path(image_path).read_bytes()).decode("utf-8"),
                                }
                            },
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": self.temperature,
                    "maxOutputTokens": self.max_tokens,
                },
            },
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()

        candidates: list[dict[str, Any]] = payload.get("candidates", [])
        if not candidates:
            return ""
        parts: list[dict[str, str]] = candidates[0].get("content", {}).get("parts", [])
        texts = [part.get("text", "").strip() for part in parts if part.get("text")]
        return "\n".join(text for text in texts if text)
        """


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
            base_url=model_config.base_url or "https://api.anthropic.com/v1",
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
