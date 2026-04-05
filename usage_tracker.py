# usage_tracker.py
import json
import threading
from datetime import datetime, timezone
from pathlib import Path

USAGE_FILE = Path.home() / ".rex" / "claude_usage.json"

# Cost per token for each provider
PROVIDER_COSTS = {
    "claude": {"input": 3.00  / 1_000_000,  "output": 15.00 / 1_000_000},
    "groq":   {"input": 0.0,                "output": 0.0},
    "gemini": {"input": 0.075 / 1_000_000,  "output": 0.30  / 1_000_000},
}

_KNOWN_PROVIDERS = ("claude", "groq", "gemini")

_lock = threading.Lock()


def _empty_provider_record() -> dict:
    return {"input_tokens": 0, "output_tokens": 0, "requests": 0, "cost_usd": 0.0}


def _empty_data() -> dict:
    return {
        "providers": {p: _empty_provider_record() for p in _KNOWN_PROVIDERS},
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }


def _load_raw() -> dict:
    """
    Load usage file, migrating from the old flat Phase-4 schema if needed.

    Old schema (no 'providers' key):
      { "total_input_tokens": N, "total_output_tokens": N,
        "total_requests": N, "estimated_cost_usd": N, "last_updated": "..." }

    New schema:
      { "providers": { "claude": {...}, "groq": {...}, "gemini": {...} },
        "last_updated": "..." }
    """
    if not USAGE_FILE.exists():
        return _empty_data()

    try:
        raw = json.loads(USAGE_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return _empty_data()

    # Migrate old flat schema — wrap existing Claude totals into new structure
    if "providers" not in raw:
        migrated = _empty_data()
        migrated["providers"]["claude"] = {
            "input_tokens":  raw.get("total_input_tokens",  0),
            "output_tokens": raw.get("total_output_tokens", 0),
            "requests":      raw.get("total_requests",      0),
            "cost_usd":      raw.get("estimated_cost_usd",  0.0),
        }
        migrated["last_updated"] = raw.get("last_updated", migrated["last_updated"])
        # Persist migrated data immediately
        USAGE_FILE.write_text(json.dumps(migrated, indent=2), encoding="utf-8")
        return migrated

    # Ensure all known providers exist (defensive — handles partial files)
    for p in _KNOWN_PROVIDERS:
        raw["providers"].setdefault(p, _empty_provider_record())

    return raw


def track_usage(provider: str, input_tokens: int, output_tokens: int) -> None:
    """Thread-safe: increment token counts and recalculate cost for a provider."""
    with _lock:
        data = _load_raw()

        if provider not in data["providers"]:
            data["providers"][provider] = _empty_provider_record()

        bucket = data["providers"][provider]
        bucket["input_tokens"]  += input_tokens
        bucket["output_tokens"] += output_tokens
        bucket["requests"]      += 1

        rates = PROVIDER_COSTS.get(provider, {"input": 0.0, "output": 0.0})
        bucket["cost_usd"] = round(
            bucket["input_tokens"]  * rates["input"] +
            bucket["output_tokens"] * rates["output"],
            6,
        )

        data["last_updated"] = datetime.now(timezone.utc).isoformat()
        USAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
        USAGE_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def get_usage() -> dict:
    """Return per-provider usage breakdown. Creates/migrates file on first call."""
    with _lock:
        return _load_raw()


def reset_usage(provider: str = None) -> dict:
    """
    Zero out usage data.

    If provider is specified, reset only that provider's bucket.
    If provider is None, reset all providers.
    Returns the full data record after reset.
    """
    with _lock:
        data = _load_raw()

        if provider:
            if provider in data["providers"]:
                data["providers"][provider] = _empty_provider_record()
        else:
            for p in list(data["providers"].keys()):
                data["providers"][p] = _empty_provider_record()

        data["last_updated"] = datetime.now(timezone.utc).isoformat()
        USAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
        USAGE_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return data
