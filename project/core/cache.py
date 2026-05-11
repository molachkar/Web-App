"""
core/cache.py
Two caching layers:

  disk  — daily JSON file for the full ML data pipeline (survives restarts)
  mem   — TTL in-memory dict for FastAPI route responses (per-process, fast)

Usage in server.py:
    from core.cache import disk, mem
    mem.get("prices") / mem.set("prices", result, ttl=30)
    disk.save(...) / disk.load()
"""

import json
import logging
import os
import threading
import time
from datetime import datetime
from typing import Any

import pandas as pd

from core.config import CACHE_FILE

log = logging.getLogger("sentinel.cache")


# ══════════════════════════════════════════════════════════════════════════════
#  DISK CACHE  —  daily ML pipeline data
# ══════════════════════════════════════════════════════════════════════════════

class _DiskCache:

    def save(self, df: pd.DataFrame, fred_ages: dict, fill_report: dict,
             fetch_log: dict, candle_note: str, fred_warnings: list) -> None:
        today = datetime.today().strftime("%Y-%m-%d")
        payload = {
            "date":          today,
            "df":            df.to_json(),
            "fred_ages":     fred_ages,
            "fill_report":   fill_report,
            "fetch_log":     fetch_log,
            "candle_note":   candle_note,
            "fred_warnings": fred_warnings,
        }
        try:
            with open(CACHE_FILE, "w") as f:
                json.dump(payload, f)
            log.info(f"Disk cache saved for {today}")
        except Exception as e:
            log.warning(f"Disk cache write failed: {e}")

    def load(self):
        """Returns tuple or None if missing/stale/corrupt."""
        if not os.path.exists(CACHE_FILE):
            return None
        try:
            with open(CACHE_FILE, "r") as f:
                payload = json.load(f)
            today = datetime.today().strftime("%Y-%m-%d")
            if payload.get("date") != today:
                os.remove(CACHE_FILE)
                log.info("Stale disk cache purged")
                return None
            df = pd.read_json(payload["df"])
            df.index = pd.to_datetime(df.index)
            df.index.name = "Date"
            df.sort_index(inplace=True)
            log.info(f"Disk cache hit — {today}")
            return (
                df,
                payload["fred_ages"],
                payload["fill_report"],
                payload["fetch_log"],
                payload["candle_note"],
                payload["fred_warnings"],
            )
        except Exception as e:
            log.warning(f"Disk cache read failed: {e}")
            try:
                os.remove(CACHE_FILE)
            except Exception:
                pass
            return None

    def invalidate(self) -> None:
        try:
            if os.path.exists(CACHE_FILE):
                os.remove(CACHE_FILE)
                log.info("Disk cache invalidated")
        except Exception as e:
            log.warning(f"Disk cache invalidation failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  MEM CACHE  —  TTL route cache for FastAPI endpoints
# ══════════════════════════════════════════════════════════════════════════════

class _MemCache:

    def __init__(self):
        self._store: dict[str, dict] = {}
        self._lock = threading.RLock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            if time.monotonic() > entry["expires_at"]:
                del self._store[key]
                return None
            return entry["value"]

    def set(self, key: str, value: Any, ttl: int = 60) -> None:
        with self._lock:
            self._store[key] = {
                "value":      value,
                "expires_at": time.monotonic() + ttl,
                "cached_at":  time.time(),
            }

    def invalidate(self, key: str | None = None) -> None:
        with self._lock:
            if key is None:
                self._store.clear()
                log.info("Mem cache fully cleared")
            elif key in self._store:
                del self._store[key]

    def stats(self) -> dict:
        with self._lock:
            now = time.monotonic()
            return {
                k: {
                    "ttl_remaining": round(max(0, v["expires_at"] - now), 1),
                    "cached_at":     v["cached_at"],
                }
                for k, v in self._store.items()
            }


# ── Singletons ────────────────────────────────────────────────────────────────
disk = _DiskCache()
mem  = _MemCache()