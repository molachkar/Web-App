"""
server.py
FastAPI entry point for Sentinel trading terminal.
Wires all routes and serves the frontend.

Endpoints:
    GET /          → serve index.html
    GET /prices    → price strip data
    GET /smc       → SMC levels
    GET /signal    → ML signal
    GET /status    → candle validator + safe window
    GET /ranges    → intraday/weekly ranges
    WS  /ws        → DOM broadcast
    GET /health    → server stats
"""

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Set

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

# Import config first
from core.config import WS_HOST, WS_PORT, TCP_HOST, TCP_PORT, FRONTEND_DIR
from core.cache import cache

# ── Logging setup ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("sentinel.server")

# ── HTML path ─────────────────────────────────────────────────
INDEX_PATH = os.path.join(FRONTEND_DIR, "index.html")

if not os.path.exists(INDEX_PATH):
    raise FileNotFoundError(f"index.html not found at: {INDEX_PATH}")

with open(INDEX_PATH, "r", encoding="utf-8") as _f:
    HTML_CONTENT = _f.read()

log.info(f"Loaded index.html from {INDEX_PATH}")

# ── WebSocket clients ─────────────────────────────────────────
clients: Set[WebSocket] = set()
stats = {"frames_rx": 0, "frames_tx": 0, "clients": 0}


# ── BROADCAST ─────────────────────────────────────────────
async def broadcast(raw: bytes):
    if not clients:
        return
    dead = set()
    await asyncio.gather(
        *[_send(ws, raw, dead) for ws in clients],
        return_exceptions=True
    )
    for ws in dead:
        clients.discard(ws)
        stats["clients"] = len(clients)


async def _send(ws, data, dead):
    try:
        await ws.send_bytes(data)
        stats["frames_tx"] += 1
    except Exception:
        dead.add(ws)


# ── MT5 TCP HANDLER ───────────────────────────────────────
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


# ── LIFESPAN ──────────────────────────────────────────────
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


# ── APP ───────────────────────────────────────────────────
app = FastAPI(title="Sentinel Terminal", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


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
    return stats


@app.get("/prices")
async def prices():
    """Return price strip data."""
    cached = cache.get("prices")
    if cached:
        return JSONResponse(cached)
    
    try:
        from data.prices import fetch_price_strip
        result = fetch_price_strip()
        cache.set("prices", result, ttl=30)
        return JSONResponse(result)
    except Exception as e:
        log.error(f"/prices error: {e}")
        return JSONResponse([], status_code=500)


@app.get("/smc")
async def smc():
    """Return SMC levels."""
    cached = cache.get("smc")
    if cached:
        return JSONResponse(cached)
    
    try:
        from smc.engine import fetch_smc_levels
        from data.prices import fetch_price_strip
        
        # Get current XAU price
        prices = fetch_price_strip()
        current = next((p["price"] for p in prices if p["symbol"] == "XAU"), 0)
        
        result = fetch_smc_levels(current)
        cache.set("smc", result, ttl=3600)
        return JSONResponse(result)
    except Exception as e:
        log.error(f"/smc error: {e}")
        return JSONResponse({}, status_code=500)


@app.get("/signal")
async def signal():
    """Return ML signal."""
    cached = cache.get("signal")
    if cached:
        # Remove non-serializable fields
        result = {k: v for k, v in cached.items() 
                  if k not in ["feat_df", "recent"]}
        return JSONResponse(result)
    
    try:
        from ml.inference import load_artefacts, run_inference
        from features.engineer import engineer_features
        from data.prices import fetch_ml_prices
        from data.fred import fetch_fred
        
        end = timedelta(days=0)
        start = timedelta(days=520)
        
        prices = fetch_ml_prices(
            datetime.utcnow() - start,
            datetime.utcnow()
        )
        fred = fetch_fred(
            datetime.utcnow() - start,
            datetime.utcnow()
        )
        
        combined = prices.join(fred, how="left").ffill().bfill()
        feat_df = engineer_features(combined)
        
        model, calib, oof = load_artefacts()
        result = run_inference(feat_df, model, calib, oof)
        
        # Cache without non-serializable fields
        cacheable = {k: v for k, v in result.items() 
                     if k not in ["feat_df", "recent"]}
        cache.set("signal", cacheable, ttl=3600)
        
        return JSONResponse(cacheable)
    except Exception as e:
        log.error(f"/signal error: {e}")
        return JSONResponse({}, status_code=500)


@app.get("/status")
async def status():
    """Return candle status and safe window info."""
    cached = cache.get("status")
    if cached:
        return JSONResponse(cached)
    
    try:
        from data.candle_validator import candle_status
        result = candle_status()
        cache.set("status", result, ttl=60)
        return JSONResponse(result)
    except Exception as e:
        log.error(f"/status error: {e}")
        return JSONResponse({}, status_code=500)


@app.get("/ranges")
async def ranges():
    """Return intraday and weekly ranges."""
    try:
        from market.ranges import fetch_ranges
        result = fetch_ranges()
        return JSONResponse(result)
    except Exception as e:
        log.error(f"/ranges error: {e}")
        return JSONResponse({}, status_code=500)


if __name__ == "__main__":
    uvicorn.run("server:app", host=WS_HOST, port=WS_PORT,
                log_level="info", access_log=False)