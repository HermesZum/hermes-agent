"""OpenCode provider profiles (Zen + Go).

Both use per-model api_mode routing:
  - OpenCode Zen: Claude → anthropic_messages, GPT-5/Codex → codex_responses,
    everything else → chat_completions (this profile)
  - OpenCode Go: GPT → codex_responses, MiniMax/Qwen → anthropic_messages,
    GLM/Kimi/DeepSeek/MiMo → chat_completions (this profile)
"""

from __future__ import annotations

from typing import Any

from hermes_cli import __version__ as _HERMES_VERSION
from providers import register_provider
from providers.base import ProviderProfile

# Attribution headers sent on every OpenCode request. Same values we send
# to OpenRouter, Vercel AI Gateway, and Fireworks. Going through
# profile.default_headers means they survive model switches and credential
# rotation. Without them OpenCode only sees the OpenAI SDK's generic
# "OpenAI/Python x.y.z" User-Agent and can't tell the traffic is Hermes Agent.
_ATTRIBUTION_HEADERS = {
    "HTTP-Referer": "https://hermes-agent.nousresearch.com",
    "X-Title": "Hermes Agent",
    "User-Agent": f"HermesAgent/{_HERMES_VERSION}",
}


def _flat_model_name(model: str | None) -> str:
    """Return the bare OpenCode model ID, tolerating aggregator prefixes."""
    return (model or "").strip().rsplit("/", 1)[-1].lower()


def _is_kimi_k2_model(model: str | None) -> bool:
    return _flat_model_name(model).startswith("kimi-k2")


def _is_deepseek_thinking_model(model: str | None) -> bool:
    m = _flat_model_name(model)
    if m.startswith("deepseek-v") and not m.startswith("deepseek-v3"):
        return True
    return m == "deepseek-reasoner"


def _is_big_pickle_model(model: str | None) -> bool:
    return _flat_model_name(model) == "big-pickle"


def _is_glm_5_2_model(model: str | None) -> bool:
    """Detect GLM-5.2 across alias spellings (glm-5.2 / glm-5-2 / glm-5p2)."""
    m = _flat_model_name(model)
    return any(token in m for token in ("glm-5.2", "glm-5-2", "glm-5p2"))


class OpenCodeGoProfile(ProviderProfile):
    """OpenCode Go - model-specific reasoning controls."""

    # Per-model completion-token cap. The opencode-go relay's default is
    # too large for mimo-v2.5-pro — it sends max_tokens=262144 but Xiaomi
    # only supports 131072 completion tokens and 400s the request.
    # Setting an explicit cap here prevents the relay default from being
    # applied. Keys are normalized via _flat_model_name().
    _MODEL_MAX_TOKENS: dict[str, int] = {
        "mimo-v2.5-pro": 131072,
    }

    def get_max_tokens(self, model: str | None) -> int | None:
        cap = self._MODEL_MAX_TOKENS.get(_flat_model_name(model))
        if cap is not None:
            return cap
        return self.default_max_tokens

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, model: str | None = None, **context
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        extra_body: dict[str, Any] = {}
        top_level: dict[str, Any] = {}

        if _is_glm_5_2_model(model):
            # GLM-5.2 on OpenCode Go uses its native OpenAI-compatible
            # reasoning_effort knob, which has exactly two enabled levels:
            # high and max. Map Hermes' richer scale onto those; leave the
            # server default alone when reasoning is disabled or unset.
            if not isinstance(reasoning_config, dict):
                return extra_body, top_level
            if reasoning_config.get("enabled") is False:
                return extra_body, top_level
            effort = (reasoning_config.get("effort") or "").strip().lower()
            if not effort or effort == "none":
                return extra_body, top_level
            top_level["reasoning_effort"] = "max" if effort in {"xhigh", "max", "ultra"} else "high"
            return extra_body, top_level

        if _is_kimi_k2_model(model):
            # Kimi K2 on OpenCode Go uses Moonshot's native wire shape:
            # extra_body.thinking (binary toggle) + top-level reasoning_effort
            # (low|medium|high). Mirrors the KimiProfile (api.moonshot.ai/v1).
            if not isinstance(reasoning_config, dict):
                # No config → leave server defaults alone.
                return extra_body, top_level

            enabled = reasoning_config.get("enabled") is not False
            if not enabled:
                extra_body["thinking"] = {"type": "disabled"}
                return extra_body, top_level

            effort = (reasoning_config.get("effort") or "").strip().lower()
            if effort in {"xhigh", "max", "ultra"}:
                top_level["reasoning_effort"] = "high"
            elif effort in {"low", "medium", "high"}:
                top_level["reasoning_effort"] = effort

            # Avoid "cannot specify both 'thinking' and 'reasoning_effort'" HTTP 400:
            # only send extra_body["thinking"] when no reasoning_effort is set.
            if "reasoning_effort" not in top_level:
                extra_body["thinking"] = {"type": "enabled"}
            return extra_body, top_level

        if not _is_deepseek_thinking_model(model):
            return extra_body, top_level

        enabled = True
        if isinstance(reasoning_config, dict) and reasoning_config.get("enabled") is False:
            enabled = False

        if not enabled:
            extra_body["thinking"] = {"type": "disabled"}
            return extra_body, top_level

        if isinstance(reasoning_config, dict):
            effort = (reasoning_config.get("effort") or "").strip().lower()
            if effort in {"xhigh", "max", "ultra"}:
                top_level["reasoning_effort"] = "max"
            elif effort in {"low", "medium", "high"}:
                top_level["reasoning_effort"] = effort

        # Avoid "cannot specify both 'thinking' and 'reasoning_effort'" HTTP 400:
        # only send extra_body["thinking"] when no reasoning_effort is set.
        if "reasoning_effort" not in top_level:
            extra_body["thinking"] = {"type": "enabled"}

        return extra_body, top_level


opencode_zen = ProviderProfile(
    name="opencode-zen",
    aliases=("opencode", "opencode_zen", "zen"),
    env_vars=("OPENCODE_ZEN_API_KEY",),
    base_url="https://opencode.ai/zen/v1",
    default_headers=dict(_ATTRIBUTION_HEADERS),
    default_aux_model="gemini-3-flash",
)

# Per-model max tokens to prevent reasoning-token truncation.
# big-pickle on OpenCode Zen has 32k max output; reasoning tokens consume
# most of this on agent prompts, leaving <1k for visible output.
# Cap at 8192 to leave ~24k for reasoning tokens.
# Apply same cap to other free-tier reasoning models on OpenCode Zen
# that are used in fallback chains or have 32k output limits.
_ocz_model_max_tokens = {
    "big-pickle": 8192,
    "nemotron-3-ultra-free": 8192,
    "nemotron-3.5-lightning-free": 8192,
    "deepseek-v4-flash-free": 8192,
    "glm-5.2": 8192,
    "glm-5.1": 8192,
    "glm-5": 8192,
    "minimax-m3": 8192,
    "minimax-m2.7": 8192,
    "minimax-m2.5": 8192,
    "kimi-k3": 8192,
    "kimi-k2.7-code": 8192,
    "kimi-k2.6": 8192,
    "kimi-k2.5": 8192,
    "qwen3.6-plus": 8192,
    "qwen3.5-plus": 8192,
    "gpt-5.6-sol": 8192,
    "gpt-5.6-terra": 8192,
    "gpt-5.6-luna": 8192,
    "gpt-5.5": 8192,
    "gpt-5.5-pro": 8192,
    "gpt-5.4": 8192,
    "gpt-5.4-pro": 8192,
    "gpt-5.4-mini": 8192,
    "gpt-5.4-nano": 8192,
    "gpt-5.3-codex-spark": 8192,
    "gpt-5.3-codex": 8192,
    "gpt-5.2": 8192,
    "gpt-5.2-codex": 8192,
    "gpt-5.1": 8192,
    "gpt-5.1-codex-max": 8192,
    "gpt-5.1-codex": 8192,
    "gpt-5.1-codex-mini": 8192,
    "gpt-5": 8192,
    "gpt-5-codex": 8192,
    "gpt-5-nano": 8192,
}

def _ocz_get_max_tokens(model: str | None) -> int | None:
    return _ocz_model_max_tokens.get(_flat_model_name(model))

opencode_zen.get_max_tokens = _ocz_get_max_tokens

opencode_go = OpenCodeGoProfile(
    name="opencode-go",
    aliases=("opencode_go", "go", "opencode-go-sub"),
    env_vars=("OPENCODE_GO_API_KEY",),
    base_url="https://opencode.ai/zen/go/v1",
    default_headers=dict(_ATTRIBUTION_HEADERS),
    default_aux_model="glm-5",
)

register_provider(opencode_zen)
register_provider(opencode_go)
