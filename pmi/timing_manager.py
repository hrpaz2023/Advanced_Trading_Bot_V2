# ------------------------------------------------------------
# File: pmi/timing_manager.py
# ------------------------------------------------------------
"""Maneja lógica de cooldown y schedule temporal."""

from __future__ import annotations

import datetime as dt


class TimingManager:
    def __init__(self):
        self._cooldowns: dict[str, dt.datetime] = {}

    def is_cooldown(self, symbol: str, now: dt.datetime) -> bool:
        until = self._cooldowns.get(symbol)
        return until is not None and now < until

    def set_cooldown(self, symbol: str, minutes: int, now: dt.datetime):
        self._cooldowns[symbol] = now + dt.timedelta(minutes=minutes)