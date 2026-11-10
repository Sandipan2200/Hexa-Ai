"""FastAPI REST + WebSocket API for Hexa."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from hexa.config import get_config
from hexa.events import Event, EventType, get_event_bus
from hexa.router import get_router
from hexa.state import State, get_state_machine
from hexa.tools import get_registry

logger = logging.getLogger(__name__)

security = HTTPBearer(auto_error=False)


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Starting Hexa API")
    yield
    logger.info("Shutting down Hexa API")


def create_app() -> FastAPI:
    """Create the FastAPI application."""
    app = FastAPI(title="Hexa API", version="0.1.0", lifespan=lifespan)

    @app.post("/command", response_model=CommandResponse, dependencies=[Depends(verify_token)])
    async def command(request: CommandRequest) -> CommandResponse:
        """Execute a text command through the router."""
        router = get_router()
        reply = router.route(request.text)
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