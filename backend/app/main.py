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

# Import and include routers
from .routes import upload, generate, refine, export
app.include_router(upload.router)
app.include_router(generate.router)
app.include_router(refine.router)
app.include_router(export.router)
