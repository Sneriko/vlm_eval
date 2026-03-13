from vlm_eval.clients import DeepSeekClient, GeminiClient, build_client
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
