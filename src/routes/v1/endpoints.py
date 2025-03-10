from fastapi import APIRouter, BackgroundTasks, WebSocket

from src.settings import Settings

settings = Settings()
router = APIRouter(prefix="/api/v1")


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    pass