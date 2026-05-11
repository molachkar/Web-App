"""
Sentinel DOM Backend
- TCP :5555  → receives JSON from MT5 EA
- WS  :8000/ws → broadcasts to browser
- GET :8000/  → serves index.html (must be in same folder as server.py)
"""

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Set

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

TCP_HOST = "0.0.0.0"
TCP_PORT = 5555
WS_HOST  = "0.0.0.0"
WS_PORT  = 8000

# Always resolve relative to THIS script file, not the cwd
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "index.html")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("sentinel.dom")

# Pre-load HTML once at startup so we catch missing file immediately
if not os.path.exists(INDEX_PATH):
    raise FileNotFoundError(f"index.html not found at: {INDEX_PATH}\nMake sure index.html is in the same folder as server.py")

with open(INDEX_PATH, "r", encoding="utf-8") as _f:
    HTML_CONTENT = _f.read()

log.info(f"Loaded index.html from {INDEX_PATH}")

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
app = FastAPI(title="Sentinel DOM", lifespan=lifespan)
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


if __name__ == "__main__":
    uvicorn.run("server:app", host=WS_HOST, port=WS_PORT,
                log_level="info", access_log=False)