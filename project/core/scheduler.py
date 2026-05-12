"""
core/scheduler.py
Background task scheduler for auto-refreshing data.
Manages periodic refreshes of signal, SMC levels, prices, and status.

Public API:
    start_scheduler()      → asyncio.Task
    get_schedule_status()  → dict
"""

import asyncio
import logging
from datetime import datetime, timedelta   # BUG 2 FIX: timedelta was missing from import
from typing import Optional

from core.config import CACHE_TTL_SECONDS, PRICES_TTL_SECONDS
from core.cache import mem                                       # BUG 3 FIX: was `cache` — doesn't exist

log = logging.getLogger("sentinel.scheduler")

_last_refresh = {
    "signal": None,
    "smc":    None,
    "prices": None,
    "status": None,
}


async def _refresh_signal():
    try:
        from ml.inference import load_artefacts, run_inference
        from features.engineer import engineer_features
        from data.prices import fetch_ml_prices
        from data.fred import fetch_fred

        log.info("Scheduler: refreshing signal...")

        end   = datetime.utcnow()
        start = end - timedelta(days=520)   # BUG 2 would have crashed here

        prices   = fetch_ml_prices(start, end)
        fred     = fetch_fred(start, end)
        combined = prices.join(fred, how="left").ffill().bfill()
        feat_df  = engineer_features(combined)

        model, calib, oof = load_artefacts()
        result = run_inference(feat_df, model, calib, oof)

        cacheable = {k: v for k, v in result.items() if k != "feat_df"}
        mem.set("signal", cacheable, ttl=CACHE_TTL_SECONDS)   # BUG 3 FIX: was cache.set

        _last_refresh["signal"] = datetime.utcnow()
        log.info("Scheduler: signal refreshed")

    except Exception as e:
        log.error(f"Scheduler: signal refresh failed: {e}")


async def _refresh_smc():
    try:
        from smc.engine import fetch_smc_levels
        from data.prices import fetch_price_strip

        log.info("Scheduler: refreshing SMC...")

        prices  = fetch_price_strip()
        current = next((p["price"] for p in prices if p["symbol"] == "XAU"), 0)
        smc     = fetch_smc_levels(current)
        mem.set("smc", smc, ttl=CACHE_TTL_SECONDS)

        _last_refresh["smc"] = datetime.utcnow()
        log.info("Scheduler: SMC refreshed")

    except Exception as e:
        log.error(f"Scheduler: SMC refresh failed: {e}")


async def _refresh_prices():
    try:
        from data.prices import fetch_price_strip
        prices = fetch_price_strip()
        mem.set("prices", prices, ttl=PRICES_TTL_SECONDS)
        _last_refresh["prices"] = datetime.utcnow()
    except Exception as e:
        log.error(f"Scheduler: prices refresh failed: {e}")


async def _refresh_status():
    try:
        from data.candle_validator import candle_status
        status = candle_status()
        mem.set("status", status, ttl=60)
        _last_refresh["status"] = datetime.utcnow()
    except Exception as e:
        log.error(f"Scheduler: status refresh failed: {e}")


async def _periodic_refresh(name: str, interval: int, coro):
    """Run a coroutine periodically."""
    await asyncio.sleep(1)
    while True:
        try:
            await coro()
        except Exception as e:
            log.error(f"Scheduler: {name} error: {e}")
        await asyncio.sleep(interval)


def start_scheduler():
    """Start all background refresh tasks."""
    async def run_all():
        await asyncio.gather(
            _periodic_refresh("prices", PRICES_TTL_SECONDS, _refresh_prices),
            _periodic_refresh("status", 60,                 _refresh_status),
            _periodic_refresh("signal", CACHE_TTL_SECONDS,  _refresh_signal),
            _periodic_refresh("smc",    CACHE_TTL_SECONDS,  _refresh_smc),
        )
    return asyncio.create_task(run_all())


def get_schedule_status() -> dict:
    """Return last refresh times for all data sources."""
    return {
        k: v.isoformat() if v else None
        for k, v in _last_refresh.items()
    }