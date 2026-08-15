"""Kilo Code provider profile."""

from providers import register_provider
from providers.base import ProviderProfile


def _flat_model_name(model: str | None) -> str:
    """Return the bare model ID, tolerating provider prefixes."""
    return (model or "").strip().rsplit("/", 1)[-1].lower()


# Per-model completion token caps for KiloCode's free-tier reasoning models.
# KiloCode's gateway relays to multiple upstreams; free models typically
# have 32k output limits that reasoning tokens consume rapidly.
# Cap at 8192 to leave ~24k for reasoning tokens.
_kilocode_model_max_tokens = {
    "nemotron-3-ultra-550b-a55b:free": 8192,
    "step-3.7-flash:free": 8192,
    "hy3:free": 8192,
    "glm-5.2": 8192,
    "minimax-m3": 8192,
    "deepseek-v4-pro-0813": 8192,
}


def _kilocode_get_max_tokens(model: str | None) -> int | None:
    return _kilocode_model_max_tokens.get(_flat_model_name(model))


kilocode = ProviderProfile(
    name="kilocode",
    aliases=("kilo-code", "kilo", "kilo-gateway"),
    env_vars=("KILOCODE_API_KEY",),
    base_url="https://api.kilo.ai/api/gateway",
    default_aux_model="google/gemini-3.6-flash",
)

kilocode.get_max_tokens = _kilocode_get_max_tokens

register_provider(kilocode)
