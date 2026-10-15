"""Model presets and provider-specific credentials shared by the UI and agents."""

import os

DEFAULT_UI_MODEL = "gpt-6-sol"
MODEL_PRESETS_BY_PROVIDER = {
    "OpenAI": (
        "gpt-6-sol", "gpt-6-astra", "gpt-6-luna",
        "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna",
    ),
    "Gemini": ("gemini-3.8-flash", "gemini-3.1-pro-preview"),
}
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def provider_for_model(model: str) -> str:
    return "Gemini" if model.strip().lower().startswith("gemini-") else "OpenAI"


def api_key_variable(model: str) -> str:
    return "GEMINI_API_KEY" if provider_for_model(model) == "Gemini" else "OPENAI_API_KEY"


def api_key_for_model(model: str, *, required: bool = True) -> str | None:
    variable = api_key_variable(model)
    value = os.environ.get(variable)
    if required and not value:
        raise RuntimeError(f"Set {variable} before using {model}.")
    return value


def llm_connection_options(
    model: str, *, api_key: str | None = None, base_url: str | None = None,
) -> dict:
    """Return SDK options without changing process-wide provider settings.

    A historical OpenAI default URL must not route a Gemini request to OpenAI.
    Explicit non-default URLs and provider-specific environment overrides allow
    an OpenAI-compatible proxy. Keys are never written to run configuration.
    """
    gemini = provider_for_model(model) == "Gemini"
    if gemini:
        if not base_url or base_url.rstrip("/") == "https://api.openai.com/v1":
            base_url = os.environ.get("GEMINI_BASE_URL") or GEMINI_BASE_URL
    else:
        if not base_url or base_url.rstrip("/") == "https://api.openai.com/v1":
            base_url = os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    return {"api_key": api_key or api_key_for_model(model), "base_url": base_url}
