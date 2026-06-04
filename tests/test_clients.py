from vlm_eval.clients import AnthropicClient, DeepSeekClient, GeminiClient, HuggingFaceClient, OpenAICompatibleClient, build_client
from vlm_eval.config import ModelConfig


def test_build_client_gemini(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    cfg = ModelConfig(
        name="gemini",
        provider="gemini",
        model="gemini-2.0-flash",
        api_key_env="GEMINI_API_KEY",
    )

    client = build_client(cfg)

    assert isinstance(client, GeminiClient)
    assert client.base_url == "https://generativelanguage.googleapis.com/v1beta"


def test_build_client_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    cfg = ModelConfig(
        name="openai",
        provider="openai_compatible",
        model="gpt-4.1-mini",
        api_key_env="OPENAI_API_KEY",
    )

    client = build_client(cfg)

    assert isinstance(client, OpenAICompatibleClient)
    assert client.base_url == "https://api.openai.com/v1"


def test_build_client_anthropic(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    cfg = ModelConfig(
        name="anthropic",
        provider="anthropic",
        model="claude-3-5-sonnet-latest",
        api_key_env="ANTHROPIC_API_KEY",
    )

    client = build_client(cfg)

    assert isinstance(client, AnthropicClient)
    assert client.base_url == "https://api.anthropic.com"


def test_build_client_deepseek(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    cfg = ModelConfig(
        name="deepseek",
        provider="deepseek",
        model="deepseek-vl2",
        api_key_env="DEEPSEEK_API_KEY",
    )

    client = build_client(cfg)

    assert isinstance(client, DeepSeekClient)
    assert client.base_url == "https://api.deepseek.com/v1"


def test_build_client_huggingface_without_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    cfg = ModelConfig(
        name="qwen-local",
        provider="huggingface",
        model="Qwen/Qwen2.5-VL-3B-Instruct",
        max_tokens=128,
        device_map="cpu",
        torch_dtype="float32",
    )

    client = build_client(cfg)

    assert isinstance(client, HuggingFaceClient)
    assert client.api_key is None
    assert client.device_map == "cpu"
    assert client.torch_dtype == "float32"


def test_build_client_huggingface_with_token(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-token")
    cfg = ModelConfig(
        name="gemma-local",
        provider="huggingface",
        model="google/gemma-3-4b-it",
        api_key_env="HF_TOKEN",
        trust_remote_code=True,
        model_kwargs={"attn_implementation": "sdpa"},
    )

    client = build_client(cfg)

    assert isinstance(client, HuggingFaceClient)
    assert client.api_key == "test-token"
    assert client.trust_remote_code is True
    assert client.model_kwargs == {"attn_implementation": "sdpa"}


def test_build_client_requires_key(monkeypatch):
    monkeypatch.delenv("MISSING_KEY", raising=False)
    cfg = ModelConfig(
        name="missing",
        provider="gemini",
        model="gemini-2.0-flash",
        api_key_env="MISSING_KEY",
    )

    try:
        build_client(cfg)
        assert False, "Expected ValueError when API key is missing"
    except ValueError as err:
        assert "MISSING_KEY" in str(err)
