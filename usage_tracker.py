# usage_tracker.py
import json
import threading
from datetime import datetime, timezone
from pathlib import Path

USAGE_FILE         = Path.home() / ".rex" / "claude_usage.json"
SONNET_INPUT_COST  = 3.00  / 1_000_000   # $3.00 per million input tokens
SONNET_OUTPUT_COST = 15.00 / 1_000_000   # $15.00 per million output tokens

_lock = threading.Lock()


def _empty_record() -> dict:
    return {
        "total_input_tokens":  0,
        "total_output_tokens": 0,
        "total_requests":      0,
        "estimated_cost_usd":  0.0,
        "last_updated":        datetime.now(timezone.utc).isoformat(),
    }


def track_usage(input_tokens: int, output_tokens: int) -> None:
    """Thread-safe: increment token counts and recalculate cost."""
    with _lock:
        if USAGE_FILE.exists():
            try:
                data = json.loads(USAGE_FILE.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, KeyError):
                data = _empty_record()
        else:
            USAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
            data = _empty_record()

        data["total_input_tokens"]  = data.get("total_input_tokens",  0) + input_tokens
        data["total_output_tokens"] = data.get("total_output_tokens", 0) + output_tokens
        data["total_requests"]      = data.get("total_requests",      0) + 1
        data["estimated_cost_usd"]  = round(
            data["total_input_tokens"]  * SONNET_INPUT_COST +
            data["total_output_tokens"] * SONNET_OUTPUT_COST,
            6,
        )
        data["last_updated"] = datetime.now(timezone.utc).isoformat()
        USAGE_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def get_usage() -> dict:
    """Return current usage totals. Returns zeroed record if file is missing."""
    if not USAGE_FILE.exists():
        return _empty_record()
    try:
        data = json.loads(USAGE_FILE.read_text(encoding="utf-8"))
        data["estimated_cost_usd"] = round(float(data.get("estimated_cost_usd", 0.0)), 6)
        return data
    except (json.JSONDecodeError, KeyError):
        return _empty_record()


def reset_usage() -> dict:
    """Zero out the usage file and return the reset record."""
    record = _empty_record()
    USAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
    USAGE_FILE.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record
