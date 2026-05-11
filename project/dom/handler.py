"""
dom/handler.py
TCP server that receives DOM data from MT5 EA and broadcasts via WebSocket.

Public API:
    start_dom_server() → asyncio.Task
    get_dom_stats() → dict
"""

import asyncio
import json
import logging
from typing import Set, Optional

from core.config import TCP_HOST, TCP_PORT

log = logging.getLogger("sentinel.dom")

# Global state
clients: Set = set()
stats = {"frames_rx": 0, "frames_tx": 0, "clients": 0}


async def broadcast(raw: bytes):
    """Broadcast raw JSON to all connected WebSocket clients."""
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
    """Send data to a single client, track failures."""
    try:
        await ws.send_bytes(data)
        stats["frames_tx"] += 1
    except Exception:
        dead.add(ws)


async def handle_mt5(reader: asyncio.StreamReader, 
                     writer: asyncio.StreamWriter):
    """Handle incoming TCP connection from MT5 EA."""
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
            
            # Validate JSON
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
    """Run the TCP server indefinitely."""
    server = await asyncio.start_server(handle_mt5, TCP_HOST, TCP_PORT)
    log.info(f"TCP listening on {TCP_HOST}:{TCP_PORT}")
    async with server:
        await server.serve_forever()


def get_dom_stats() -> dict:
    """Return current DOM statistics."""
    return stats.copy()
