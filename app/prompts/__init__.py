"""Prompt loader for LLM system messages.

All prompts live as YAML files in this directory.  Each file has:

    name: short_name
    description: >
      Human-readable purpose.
    includes:            # optional — list of shared section names
      - special_characters
      - extraction_errors
    system_message: |
      Full prompt text ...

Shared sections live in ``_shared.yaml`` as named YAML keys. When a prompt
lists ``includes``, those sections are appended (in order) to the end of
its ``system_message`` before it is returned.

Usage::

    from app.prompts import render_prompt

    # Static prompt (no placeholders)
    msg = render_prompt("conflict_batch")

    # Dynamic prompt (with {variable} placeholders)
    msg = render_prompt("rescue_numeric", context_display=ctx, seg_id=sid, ...)
"""

from __future__ import annotations

from pathlib import Path

import yaml

_PROMPTS_DIR = Path(__file__).resolve().parent
_cache: dict[str, dict] = {}
_shared_cache: dict[str, str] | None = None


class _SafeFormatDict(dict):
    """Dict subclass that returns '{key}' for missing keys instead of raising KeyError.

    This lets dynamic prompts be partially rendered — any placeholder without
    a matching kwarg is left as-is in the output string.
    """

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _load_shared() -> dict[str, str]:
    """Load the shared sections file, caching for the process lifetime."""
    global _shared_cache
    if _shared_cache is not None:
        return _shared_cache

    path = _PROMPTS_DIR / "_shared.yaml"
    if not path.exists():
        _shared_cache = {}
        return _shared_cache

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    _shared_cache = {k: v for k, v in (data or {}).items() if isinstance(v, str)}
    return _shared_cache


def load_prompt(name: str) -> dict:
    """Load a prompt YAML file by name, returning the full dict.

    If the prompt declares ``includes: [section_name, ...]``, the named
    sections from ``_shared.yaml`` are appended to ``system_message``.

    Results are cached for the lifetime of the process.
    """
    if name in _cache:
        return _cache[name]

    path = _PROMPTS_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or "system_message" not in data:
        raise ValueError(f"Prompt file {path} must contain a 'system_message' key")

    # Append shared sections if requested
    includes = data.get("includes") or []
    if includes:
        shared = _load_shared()
        parts = [data["system_message"].rstrip("\n")]
        for section_name in includes:
            section_text = shared.get(section_name)
            if section_text is None:
                raise ValueError(
                    f"Prompt {name!r} includes unknown shared section {section_name!r}. "
                    f"Available: {sorted(shared.keys())}"
                )
            parts.append(section_text.rstrip("\n"))
        data["system_message"] = "\n\n".join(parts) + "\n"

    _cache[name] = data
    return data


def render_prompt(name: str, **kwargs: str) -> str:
    """Load a prompt and return its system_message, optionally formatted.

    If kwargs are provided, ``str.format_map`` is applied with a safe dict
    that leaves unknown ``{placeholders}`` intact rather than crashing.
    If no kwargs, the raw system_message is returned as-is (no format call,
    so literal braces in static prompts are safe).
    """
    data = load_prompt(name)
    msg = data["system_message"]
    if kwargs:
        msg = msg.format_map(_SafeFormatDict(kwargs))
    return msg


def clear_cache() -> None:
    """Clear the in-memory prompt cache (useful for testing)."""
    global _shared_cache
    _cache.clear()
    _shared_cache = None
