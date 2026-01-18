"""
backend/app/main.py

FastAPI Entrypoint.
Orchestrates the entire Glimpse3D backend pipeline.

Responsibilities:
- Initialize FastAPI app
- Register routers (upload, generate, refine, export, status)
- Setup middleware (CORS, logging)
- Health check endpoints
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import upload, generate, refine, export, status
from app.core.logger import logger

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
    return {
        "message": "Glimpse3D Backend is running",
        "version": "0.1.0",
        "status": "healthy"
    }

@app.get("/health")
async def health_check():
    """Detailed health check with Supabase connection test."""
    try:
        from app.core.supabase_client import get_supabase
        supabase = get_supabase()
        # Simple query to test connection
        supabase.table("projects").select("id").limit(1).execute()
        db_status = "connected"
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}")
        db_status = f"error: {str(e)}"
    
    return {
        "status": "healthy",
        "database": db_status,
        "version": "0.1.0"
    }

# Include routers
app.include_router(upload.router, prefix="/api/v1")
app.include_router(generate.router, prefix="/api/v1")
app.include_router(refine.router, prefix="/api/v1")
app.include_router(export.router, prefix="/api/v1")
app.include_router(status.router, prefix="/api/v1")

# Import and include compare router
from app.routes import compare
app.include_router(compare.router, prefix="/api/v1")

logger.info("Glimpse3D Backend initialized successfully")
