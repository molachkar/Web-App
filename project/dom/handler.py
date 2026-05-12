"""
dom/handler.py
DOM TCP server — receives JSON frames from MT5 EA and broadcasts to WebSocket clients.

Extracted from server.py so server.py stays routes-only.

BUG 4 FIX: this file previously contained a copy of smc/engine.py (fetch_smc_levels,
            _fetch_4h, etc.) — none of the required DOM symbols existed.
BUG 5 FIX: get_dom_stats() was imported in dom/__init__.py but never defined anywhere.

Public API:
    broadcast(raw)     → coroutine, sends bytes to all connected WS clients
    tcp_server()       → coroutine, starts the MT5 TCP listener on TCP_HOST:TCP_PORT
    get_dom_stats()    → dict snapshot of live counters
    clients            → set of active WebSocket connections (shared with server.py)
    stats              → live frame/client counters dict (shared with server.py)
"""

import asyncio
import json
import logging
from typing import Set

from fastapi import WebSocket

log = logging.getLogger("sentinel.dom")

# ── Shared state ──────────────────────────────────────────────────────────────
# server.py imports these directly so both the WS endpoint and the TCP handler
# operate on the same set of clients and the same counters.
clients: Set[WebSocket] = set()
stats = {"frames_rx": 0, "frames_tx": 0, "clients": 0}


# ── Broadcast ─────────────────────────────────────────────────────────────────
async def broadcast(raw: bytes) -> None:
    """Send raw bytes to every connected WebSocket client."""
    if not clients:
        return
    dead = set()
    await asyncio.gather(
        *[_send(ws, raw, dead) for ws in clients],
        return_exceptions=True,
    )
    for ws in dead:
        clients.discard(ws)
        stats["clients"] = len(clients)


async def _send(ws: WebSocket, data: bytes, dead: set) -> None:
    try:
        await ws.send_bytes(data)
        stats["frames_tx"] += 1
    except Exception:
        dead.add(ws)


# ── MT5 TCP handler ───────────────────────────────────────────────────────────
async def handle_mt5(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> None:
    addr = writer.get_extra_info("peername")
    log.info(f"MT5 connected from {addr}")
    try:
        while True:
            line = await reader.readline()
            if not line:
                break
            line = line.strip()
            if not line or line[0] != ord(b"{"):
                continue
            try:
                json.loads(line)          # validate JSON before broadcasting
            except Exception:
                continue
            stats["frames_rx"] += 1
            await broadcast(line)
    except Exception as e:
        log.warning(f"MT5 connection error: {e}")
    finally:
        writer.close()
        log.info(f"MT5 disconnected: {addr}")


async def tcp_server() -> None:
    """Start the TCP listener. Called from server.py lifespan."""
    from core.config import TCP_HOST, TCP_PORT
    server = await asyncio.start_server(handle_mt5, TCP_HOST, TCP_PORT)
    log.info(f"TCP listening on {TCP_HOST}:{TCP_PORT}")
    async with server:
        await server.serve_forever()


# ── Stats (BUG 5 FIX) ─────────────────────────────────────────────────────────
def get_dom_stats() -> dict:
    """
    Return a snapshot of live DOM counters.
    Called by /health endpoint and dom/__init__.py.
    """
    return dict(stats)