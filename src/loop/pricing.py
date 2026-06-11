"""USD cost estimation. Tokens are the authoritative unit; USD is a convenience
derived from a pricing table that carries an as-of date.
"""
from __future__ import annotations

import warnings
from datetime import date, datetime
from typing import Any

STALENESS_DAYS = 90

DEFAULT_PRICING: dict[str, Any] = {
    "as_of": "2026-01-15",
    "models": {
        # USD per million tokens
        "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
        "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
        "claude-opus-4-8": {"input": 15.00, "output": 75.00},
        "gpt-4o": {"input": 2.50, "output": 10.00},
    },
}


class PricingStalenessError(Exception):
    pass


def check_staleness(pricing: dict[str, Any], behavior: str = "warn", today: date | None = None) -> bool:
    """Returns True if cost accounting should proceed in USD; False for token-only fallback."""
    as_of = datetime.strptime(pricing["as_of"], "%Y-%m-%d").date()
    age = ((today or date.today()) - as_of).days
    if age <= STALENESS_DAYS:
        return True
    msg = f"pricing table as_of={pricing['as_of']} is {age} days old (> {STALENESS_DAYS})"
    if behavior == "refuse":
        raise PricingStalenessError(msg + "; refusing USD cost accounting")
    if behavior == "token-only":
        return False
    warnings.warn(msg + "; USD figures are estimates — tokens remain authoritative", stacklevel=2)
    return True


def estimate_cost(input_tokens: int, output_tokens: int, model_id: str, pricing: dict[str, Any]) -> float | None:
    rates = pricing["models"].get(model_id)
    if not rates:
        return None
    return (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000
