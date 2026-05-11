"""
server.py
FastAPI entry point for Sentinel trading terminal.

Endpoints:
    GET  /         → serve index.html
    GET  /prices   → price strip
    GET  /smc      → SMC levels (4H)
    GET  /signal   → ML signal
    GET  /status   → candle validator + safe window
    GET  /ranges   → intraday / weekly ranges
    GET  /health   → server stats + cache status
    WS   /ws       → DOM broadcast
    POST /cache/invalidate → force-clear mem cache
"""

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Set

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from core.config import (
    WS_HOST, WS_PORT, TCP_HOST, TCP_PORT,
    FRONTEND_DIR, DAYS_BACK,
)
from core.cache import disk, mem   # disk = daily pipeline cache, mem = route TTL cache

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sentinel.server")

# ── HTML ───────────────────────────────────────────────────────────────────────
INDEX_PATH = os.path.join(FRONTEND_DIR, "index.html")

if not os.path.exists(INDEX_PATH):
    raise FileNotFoundError(f"index.html not found at: {INDEX_PATH}")

with open(INDEX_PATH, "r", encoding="utf-8") as _f:
    HTML_CONTENT = _f.read()

log.info(f"Loaded index.html from {INDEX_PATH}")

# ── WebSocket state ────────────────────────────────────────────────────────────
clients: Set[WebSocket] = set()
stats = {"frames_rx": 0, "frames_tx": 0, "clients": 0}


# ── Broadcast ──────────────────────────────────────────────────────────────────
async def broadcast(raw: bytes):
    if not clients:
        return
    dead = set()
    await asyncio.gather(*[_send(ws, raw, dead) for ws in clients], return_exceptions=True)
    for ws in dead:
        clients.discard(ws)
        stats["clients"] = len(clients)


async def _send(ws, data, dead):
    try:
        await ws.send_bytes(data)
        stats["frames_tx"] += 1
    except Exception:
        dead.add(ws)


# ── MT5 TCP handler ────────────────────────────────────────────────────────────
async def handle_mt5(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    addr = writer.get_extra_info("peername")
    log.info(f"MT5 connected from {addr}")
    try:
        while True:
            line = await reader.readline()
            if not line:
                break
            line = line.strip()
            if not line or line[0] != ord(b'{'):
                continue
            try:
                json.loads(line)
            except Exception:
                continue
            stats["frames_rx"] += 1
            await broadcast(line)
    except Exception as e:
        log.warning(f"MT5 connection error: {e}")
    finally:
        writer.close()
        log.info(f"MT5 disconnected: {addr}")


async def tcp_server():
    server = await asyncio.start_server(handle_mt5, TCP_HOST, TCP_PORT)
    log.info(f"TCP listening on {TCP_HOST}:{TCP_PORT}")
    async with server:
        await server.serve_forever()


# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(tcp_server())
    log.info(f"WS endpoint → ws://localhost:{WS_PORT}/ws")
    log.info(f"Dashboard  → http://localhost:{WS_PORT}/")
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Sentinel Terminal", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
async def index():
    return HTMLResponse(content=HTML_CONTENT)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    clients.add(ws)
    stats["clients"] = len(clients)
    log.info(f"WS client connected | active={len(clients)}")
    try:
        while True:
            try:
                msg = await asyncio.wait_for(ws.receive_text(), timeout=30.0)
                if msg == "ping":
                    await ws.send_text('{"type":"pong"}')
            except asyncio.TimeoutError:
                await ws.send_text('{"type":"keepalive"}')
    except WebSocketDisconnect:
        pass
    except Exception as e:
        log.warning(f"WS error: {e}")
    finally:
        clients.discard(ws)
        stats["clients"] = len(clients)
        log.info(f"WS client disconnected | active={len(clients)}")


@app.get("/health")
async def health():
    return {**stats, "cache": mem.stats()}


@app.get("/prices")
async def prices():
    cached = mem.get("prices")
    if cached is not None:
        return JSONResponse(cached)
    try:
        from data.prices import fetch_price_strip
        result = fetch_price_strip()
        mem.set("prices", result, ttl=30)
        return JSONResponse(result)
    except Exception as e:
        log.error(f"/prices error: {e}")
        return JSONResponse([], status_code=500)


@app.get("/smc")
async def smc():
    cached = mem.get("smc")
    if cached is not None:
        return JSONResponse(cached)
    try:
        from smc.engine import fetch_smc_levels
        from data.prices import fetch_price_strip

        strip   = fetch_price_strip()
        current = next((p["price"] for p in strip if p["symbol"] == "XAU" and p["price"]), 0.0)
        result  = fetch_smc_levels(float(current))
        mem.set("smc", result, ttl=3600)
        return JSONResponse(result)
    except Exception as e:
        log.error(f"/smc error: {e}")
        return JSONResponse({}, status_code=500)


@app.get("/signal")
async def signal():
    cached = mem.get("signal")
    if cached is not None:
        return JSONResponse(cached)
    try:
        from data.prices import fetch_ml_prices
        from data.fred import fetch_fred
        from features.engineer import engineer_features
        from ml.inference import load_artefacts, run_inference

        end   = datetime.utcnow()
        start = end - timedelta(days=DAYS_BACK)

        prices  = fetch_ml_prices(start, end)
        macro   = fetch_fred(start, end)
        combined = prices.join(macro, how="left").ffill().bfill()
        feat_df  = engineer_features(combined)

        model, calib, oof = load_artefacts()
        result = run_inference(feat_df, model, calib, oof)

        mem.set("signal", result, ttl=3600)
        return JSONResponse(result)
    except Exception as e:
        log.error(f"/signal error: {e}")
        return JSONResponse({"signal": "ERROR", "error": str(e)}, status_code=500)


@app.get("/status")
async def status():
    cached = mem.get("status")
    if cached is not None:
        return JSONResponse(cached)
    try:
        from data.candle_validator import candle_status
        result = candle_status()
        mem.set("status", result, ttl=60)
        return JSONResponse(result)
    except Exception as e:
        log.error(f"/status error: {e}")
        return JSONResponse({}, status_code=500)


@app.get("/ranges")
async def ranges():
    # Short TTL — ranges change every few minutes
    cached = mem.get("ranges")
    if cached is not None:
        return JSONResponse(cached)
    try:
        from market.ranges import fetch_ranges
        result = fetch_ranges()
        mem.set("ranges", result, ttl=120)
        return JSONResponse(result)
    except Exception as e:
        log.error(f"/ranges error: {e}")
        return JSONResponse({}, status_code=500)


@app.post("/cache/invalidate")
async def invalidate_cache():
    """Force-clear all in-memory route caches."""
    mem.invalidate()
    return {"status": "ok", "message": "All route caches cleared"}


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host=WS_HOST, port=WS_PORT,
        log_level="info", access_log=False,
    )