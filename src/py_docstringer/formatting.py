"""Small text formatting helpers."""

from __future__ import annotations


def pluralize(count: int, singular: str, plural: str) -> str:
    """Return the singular word when count is one, otherwise the plural word."""

    return singular if count == 1 else plural
