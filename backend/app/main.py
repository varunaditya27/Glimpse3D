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

# TODO: Import and include routers
# from app.routes import upload, generate, refine, export
# app.include_router(upload.router)
# app.include_router(generate.router)
# app.include_router(refine.router)
# app.include_router(export.router)
