"""
backend/app/main.py

FastAPI Entrypoint.
Orchestrates the entire Glimpse3D backend pipeline.

Responsibilities:
- Initialize FastAPI app
- Register routers (upload, generate, refine, export)
- Setup middleware (CORS, logging)
- Health check endpoints
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Glimpse3D Backend",
    description="API for Single-Image 3D Reconstruction & Enhancement",
    version="0.1.0"
)

from fastapi.staticfiles import StaticFiles
from pathlib import Path

# Mount static directories
# Go up to project root
project_root = Path(__file__).parent.parent.parent
output_dir = project_root / "assets" / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

app.mount("/outputs", StaticFiles(directory=str(output_dir)), name="outputs")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "Glimpse3D Backend is running"}

# Import routers
from .routes import upload, generate, refine, export

# WebSocket Endpoint
from fastapi import WebSocket, WebSocketDisconnect
from .services.websocket_manager import manager

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            # Keep connection alive and listen for client commands if needed
            # For now, we mainly push updates from server
            data = await websocket.receive_text()
            # echo or handle commands
            # await manager.send_personal_message({"message": "Ack"}, client_id)
    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        # logger.error(f"WebSocket error: {e}")
        manager.disconnect(client_id)

app.include_router(upload.router)
app.include_router(generate.router)
app.include_router(refine.router)
app.include_router(export.router)

