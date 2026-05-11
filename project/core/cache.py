"""
core/cache.py
Simple JSON-based cache with per-key TTL.
Used to avoid hammering yfinance / FRED on every page load.

Usage:
    from core.cache import cache
    cache.set("signal", payload, ttl=3600)
    result = cache.get("signal")   # None if expired or missing
"""

import json
import os
import time
import threading
import logging
from typing import Any, Optional

from core.config import ROOT_DIR, CACHE_TTL_SECONDS

log = logging.getLogger("sentinel.cache")

CACHE_FILE = os.path.join(ROOT_DIR, ".cache.json")


class DiskCache:
    """
    Thread-safe, TTL-aware JSON cache backed by a single flat file.

    Each entry is stored as:
        { "value": <any JSON-serialisable>, "expires_at": <unix timestamp> }

    The in-memory dict is the source of truth during a run; the file is
    written on every set() and read on startup so data survives restarts
    within the same calendar day.
    """

    def __init__(self, path: str = CACHE_FILE):
        self._path = path
        self._lock = threading.Lock()
        self._store: dict[str, dict] = {}
        self._load()

    # ── private ───────────────────────────────────────────────────────────────
    def _load(self):
        if os.path.exists(self._path):
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    self._store = json.load(f)
                log.debug(f"Cache loaded from {self._path} ({len(self._store)} keys)")
            except Exception as exc:
                log.warning(f"Cache file unreadable, starting fresh: {exc}")
                self._store = {}

    def _save(self):
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(self._store, f, default=str)
        except Exception as exc:
            log.warning(f"Cache write failed: {exc}")

    def _is_valid(self, entry: dict) -> bool:
        return time.time() < entry.get("expires_at", 0)

    # ── public ────────────────────────────────────────────────────────────────
    def get(self, key: str) -> Optional[Any]:
        """Return cached value if present and not expired, else None."""
        with self._lock:
            entry = self._store.get(key)
            if entry and self._is_valid(entry):
                log.debug(f"Cache HIT  [{key}]")
                return entry["value"]
            if entry:
                log.debug(f"Cache MISS [{key}] (expired)")
            return None

    def set(self, key: str, value: Any, ttl: int = CACHE_TTL_SECONDS) -> None:
        """Store value under key, expiring after ttl seconds."""
        with self._lock:
            self._store[key] = {
                "value":      value,
                "expires_at": time.time() + ttl,
                "cached_at":  time.time(),
            }
            self._save()
        log.debug(f"Cache SET  [{key}] ttl={ttl}s")

    def delete(self, key: str) -> None:
        """Remove a key from the cache."""
        with self._lock:
            self._store.pop(key, None)
            self._save()

    def invalidate_all(self) -> None:
        """Wipe the entire cache (force full refresh on next request)."""
        with self._lock:
            self._store.clear()
            self._save()
        log.info("Cache invalidated (all keys cleared)")

    def age(self, key: str) -> Optional[float]:
        """Return seconds since this key was cached, or None if missing/expired."""
        with self._lock:
            entry = self._store.get(key)
            if entry and self._is_valid(entry):
                return time.time() - entry.get("cached_at", time.time())
            return None

    def status(self) -> dict:
        """Return a summary dict for the /status endpoint."""
        now = time.time()
        out = {}
        with self._lock:
            for k, v in self._store.items():
                expires_in = v.get("expires_at", 0) - now
                out[k] = {
                    "valid":      expires_in > 0,
                    "expires_in": round(max(expires_in, 0)),
                    "age":        round(now - v.get("cached_at", now)),
                }
        return out


# Module-level singleton — import and use directly
cache = DiskCache()