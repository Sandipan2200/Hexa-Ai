"""FastAPI REST + WebSocket API for Hexa."""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

import hexa
from hexa.config import get_config
from hexa.events import Event, EventType, get_event_bus
from hexa.router import get_router
from hexa.state import State, get_state_machine
from hexa.tools import get_registry

logger = logging.getLogger(__name__)

security = HTTPBearer(auto_error=False)

# Set in create_app; used by the /health uptime readout.
_app_started_at: float | None = None


class CommandRequest(BaseModel):
    text: str


class CommandResponse(BaseModel):
    reply: str


class StateResponse(BaseModel):
    state: str


class ToolsResponse(BaseModel):
    tools: list[dict[str, Any]]


class ConfigResponse(BaseModel):
    config: dict[str, Any]


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify bearer token."""
    config = get_config()
    if not credentials or credentials.credentials != config.api.bearer_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing bearer token",
        )
    return credentials.credentials


_INDEX_HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Hexa - Offline Voice Assistant</title>
  <style>
    body {{
      font-family: "Segoe UI", system-ui, sans-serif;
      background: #111418; color: #d7dce2; margin: 0;
      display: grid; place-items: center; min-height: 100vh;
    }}
    .card {{
      max-width: 560px; padding: 2.5rem; border-radius: 12px;
      background: #1a1f26; box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4);
    }}
    h1 {{ color: #7cb2ff; margin: 0 0 0.25rem; font-size: 1.6rem; }}
    .sub {{ color: #8b94a1; margin-bottom: 1.25rem; }}
    ul {{ line-height: 1.8; padding-left: 1.1rem; }}
    a {{ color: #7cb2ff; }}
    code {{
      background: #232a33; padding: 0.1rem 0.35rem; border-radius: 4px;
      font-size: 0.92em;
    }}
  </style>
</head>
<body>
  <div class="card">
    <h1>Hexa is running</h1>
    <p class="sub">100% offline voice assistant core &middot; v{hexa.__version__}</p>
    <ul>
      <li><a href="/health" rel="noopener"><code>/health</code></a> - liveness status (JSON)</li>
      <li><a href="/docs" rel="noopener"><code>/docs</code></a> - interactive API docs</li>
      <li><code>/command</code>, <code>/state</code>, <code>/tools</code>, <code>/config</code> - require the bearer token from <code>config.yaml</code></li>
    </ul>
    <p class="sub">REST and WebSocket endpoints authenticate with the bearer token
    (<code>Authorization: Bearer &lt;token&gt;</code> / <code>?token=</code> for events).</p>
  </div>
</body>
</html>
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Starting Hexa API")
    yield
    logger.info("Shutting down Hexa API")


def create_app() -> FastAPI:
    """Create the FastAPI application."""
    global _app_started_at
    _app_started_at = time.time()
    app = FastAPI(title="Hexa API", version=hexa.__version__, lifespan=lifespan)

    # ------------------------------------------------------------------
    # Unauthenticated info/liveness endpoints.
    # Browsers and uptime probes hitting the service root previously got a
    # bare 404 that looked like a fault. These endpoints expose nothing
    # sensitive (no state, no config, no token) - all real endpoints below
    # still require the bearer token. Interactive docs remain at /docs.
    # ------------------------------------------------------------------
    @app.get("/", include_in_schema=False)
    async def index() -> HTMLResponse:
        return HTMLResponse(_INDEX_HTML)

    @app.get("/health", include_in_schema=False)
    async def health() -> dict[str, Any]:
        uptime = time.time() - _app_started_at if _app_started_at else 0.0
        return {
            "status": "ok",
            "service": "hexa",
            "version": hexa.__version__,
            "uptime_seconds": round(uptime, 1),
        }

    @app.post("/command", response_model=CommandResponse, dependencies=[Depends(verify_token)])
    async def command(request: CommandRequest) -> CommandResponse:
        """Execute a text command through the router."""
        try:
            reply = get_router().route(request.text)
        except Exception as e:
            # A failed chat/tool request must never take the service down.
            # The router already converts LLM/tool failures to readable text;
            # this is the final safety net for unexpected errors.
            logger.exception("Unhandled error processing command")
            reply = f"Error: {e}"
        return CommandResponse(reply=reply)

    @app.get("/state", response_model=StateResponse, dependencies=[Depends(verify_token)])
    async def get_state() -> StateResponse:
        """Get current state."""
        sm = get_state_machine()
        return StateResponse(state=sm.current_state.value)

    @app.get("/tools", response_model=ToolsResponse, dependencies=[Depends(verify_token)])
    async def get_tools() -> ToolsResponse:
        """Get available tools with schemas."""
        registry = get_registry()
        return ToolsResponse(tools=registry.get_schemas())

    @app.get("/config", response_model=ConfigResponse, dependencies=[Depends(verify_token)])
    async def get_config_endpoint() -> ConfigResponse:
        """Get configuration (secrets hidden)."""
        config = get_config()
        data = config.model_dump(mode="python")
        # Hide bearer token
        if "api" in data and "bearer_token" in data["api"]:
            data["api"]["bearer_token"] = "***" if data["api"]["bearer_token"] else ""
        return ConfigResponse(config=data)

    @app.post("/wake", dependencies=[Depends(verify_token)])
    async def wake() -> dict[str, str]:
        """Simulate wake word detection."""
        get_event_bus().emit(Event(type=EventType.WAKE_DETECTED, data={}))
        return {"status": "wake simulated"}

    @app.post("/stop", dependencies=[Depends(verify_token)])
    async def stop() -> dict[str, str]:
        """Stop speaking / cancel current operation."""
        get_event_bus().emit(Event(type=EventType.INTERRUPT, data={"reason": "api"}))
        return {"status": "stopped"}

    @app.websocket("/events")
    async def events_websocket(websocket: WebSocket, token: str | None = None):
        """WebSocket for real-time events."""
        config = get_config()
        if token != config.api.bearer_token:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

        await websocket.accept()
        bus = get_event_bus()

        try:
            while True:
                event = await bus.get_event()
                await websocket.send_json({
                    "type": event.type.value,
                    "data": event.data,
                    "timestamp": event.timestamp.isoformat(),
                })
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected")
        except Exception as e:
            logger.exception("WebSocket error: %s", e)
            await websocket.close()

    return app