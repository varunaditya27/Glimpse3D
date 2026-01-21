"""
Handles the initial coarse 3D generation.

Responsibilities:
- Accept processed image ID
- Trigger TripoSR or LGM model inference
- Generate initial Gaussian Splat (.ply or .splat)
- Return 3D model URL for frontend viewer
"""

import os
import tempfile
from pathlib import Path
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional

from ..core.logger import get_logger
from ..services.pipeline_manager import PipelineManager

logger = get_logger(__name__)
router = APIRouter(prefix="/generate", tags=["Generate"])

# Global pipeline manager instance
pipeline_manager = PipelineManager()

class GenerateRequest(BaseModel):
    image_path: str
    output_dir: Optional[str] = None

class GenerateResponse(BaseModel):
    success: bool
    status: str
    model_url: Optional[str] = None
    error: Optional[str] = None
    job_id: Optional[str] = None

# Store running jobs (in production, use Redis/database)
running_jobs = {}

@router.post("/")
async def generate_3d(request: GenerateRequest, background_tasks: BackgroundTasks):
    """
    Triggers the coarse 3D reconstruction from the uploaded image.
    """
    try:
        # Validate input
        if not os.path.exists(request.image_path):
            raise HTTPException(status_code=400, detail=f"Image not found: {request.image_path}")

        # Create output directory
        # Generate unique job ID
        job_id = f"gen_{hash(request.image_path) % 1000000}"

        # Create output directory
        output_dir = request.output_dir
        if not output_dir:
            # Use persistent assets/outputs directory
            from ..core.config import settings
            output_dir = settings.PROJECT_ROOT / "assets" / "outputs" / job_id
            
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Store job status
        running_jobs[job_id] = {
            'status': 'starting',
            'progress': 0.0,
            'result': None,
            'error': None,
            'warnings': [],
            'start_time': time.time()
        }

        # Add state callback to track progress
        # We define this as an async function if we want to await websocket send,
        # but PipelineManager callbacks are synchronous.
        # So we need to schedule the async send on the event loop.
        
        from ..services.websocket_manager import manager
        
        def update_job_status(state):
            # Update local state
            running_jobs[job_id]['status'] = state.stage.value
            running_jobs[job_id]['progress'] = state.progress
            if state.error:
                running_jobs[job_id]['error'] = state.error
            if state.warnings:
                running_jobs[job_id]['warnings'] = state.warnings
            
            # Prepare message
            msg = {
                "type": "progress_update",
                "job_id": job_id,
                "status": state.stage.value,
                "progress": state.progress,
                "message": state.message,
                "error": state.error,
                "warnings": state.warnings
            }
            
            # If completed, include result
            if state.stage.value == "completed":
                 # We need to fetch the final result which is constructed in run_generation_task
                 # Alternatively, we can let run_generation_task handle the final "completed" emit.
                 # But state callback receives "completed" state too.
                 # The PipelineResult holds the path, but the "state" object might just hold message.
                 # Let's rely on run_generation_task for the FINAL detailed message with URL, 
                 # and use this for progress.
                 pass

            # Schedule async send
            try:
                loop = asyncio.get_running_loop()
                # We broadcast to the specific job_id if we treat job_id as client_id 
                # OR we assume client subscribed with some ID. 
                # Ideally, client connects with a session ID, and we send to that session.
                # For simplicity here: The frontend uses a fixed ID or we just broadcast to all?
                # Better: The frontend connects with a distinct Client ID.
                # But we don't know WHICH client triggered this job here easily unless we pass Client-ID in request.
                # Let's assume for this MVP we BROADCAST to all connected clients (single user mode) 
                # OR we update the generate request to accept a client_id.
                
                # Using broadcast for MVP reliability in single-user scenario
                loop.create_task(manager.broadcast(msg))
            except RuntimeError:
                pass # No loop

        pipeline_manager.add_state_callback(update_job_status)

        # Run generation in background
        background_tasks.add_task(run_generation_task, job_id, request.image_path, output_dir)
        
        # Ensure cleanup loop is running
        ensure_cleanup_loop()

        return GenerateResponse(
            success=True,
            status="generation_started",
            job_id=job_id
        )

    except Exception as e:
        logger.error(f"Generation request failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

import asyncio
import time

# Timeout Manager
CLEANUP_task = None

async def job_timeout_monitor():
    """Periodically check for stale jobs."""
    logger.info("Starting Job Timeout Monitor...")
    while True:
        try:
            now = time.time()
            timeout = 600  # 10 Minutes
            
            # List because we might modify dictionary
            for job_id, job in list(running_jobs.items()):
                if job['status'] in ['starting', 'processing']:
                    start_time = job.get('start_time')
                    if start_time and (now - start_time > timeout):
                        logger.warning(f"Killing stale job {job_id} (Duration: {now-start_time:.1f}s)")
                        running_jobs[job_id]['status'] = 'failed'
                        running_jobs[job_id]['error'] = 'Job timed out (10 min limit)'
                        # Note: We can't easily kill the asyncio task itself unless we stored the Task object
                        # But marking it failed stops the frontend polling.
            
            await asyncio.sleep(60) # Check every minute
        except Exception as e:
            logger.error(f"Timeout monitor error: {e}")
            await asyncio.sleep(60)

def ensure_cleanup_loop():
    global CLEANUP_task
    try:
        if CLEANUP_task is None or CLEANUP_task.done():
            loop = asyncio.get_running_loop()
            CLEANUP_task = loop.create_task(job_timeout_monitor())
    except RuntimeError:
        pass # No loop running

@router.get("/status/{job_id}")
async def get_generation_status(job_id: str):
    """Get the status of a generation job."""
    if job_id not in running_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = running_jobs[job_id]
    return {
        'job_id': job_id,
        'status': job['status'],
        'progress': job['progress'],
        'result': job.get('result'),
        'error': job.get('error'),
        'warnings': job.get('warnings', [])
    }

async def run_generation_task(job_id: str, image_path: str, output_dir: str):
    """Background task to run the generation pipeline."""
    try:
        logger.info(f"Starting generation task {job_id} for {image_path}")

        # Run FULL pipeline
        result = await pipeline_manager.run_pipeline(image_path, output_dir)

        if result.success and result.final_model_path:
            # Convert absolute file path to HTTP URL
            # Backend serves files at /outputs/*, so we need relative path from assets/outputs/
            try:
                from ..core.config import settings
                outputs_dir = settings.PROJECT_ROOT / "assets" / "outputs"
                model_path = Path(result.final_model_path)

                # Get relative path from outputs directory
                if model_path.is_absolute():
                    try:
                        relative_path = model_path.relative_to(outputs_dir)
                        model_url = f"/outputs/{relative_path}".replace("\\", "/")
                    except ValueError:
                        # If not under outputs dir, just use filename
                        model_url = f"/outputs/{job_id}/{model_path.name}".replace("\\", "/")
                else:
                    model_url = f"/outputs/{model_path}"

                logger.info(f"Model URL: {model_url} (from {result.final_model_path})")

            except Exception as e:
                logger.error(f"Failed to construct model URL: {e}")
                model_url = f"/outputs/{job_id}/model.ply"  # Fallback

            running_jobs[job_id].update({
                'status': 'completed',
                'progress': 1.0,
                'result': {
                    'model_url': model_url,
                    'model_type': 'ply'
                },
                'warnings': result.warnings
            })

            logger.info(f"Generation task {job_id} completed successfully")
            
            # Notify WebSocket
            try:
                from ..services.websocket_manager import manager
                # Use asyncio.create_task since we are in async context
                msg = {
                    "type": "job_completed",
                    "job_id": job_id,
                    "model_url": model_url,
                    "status": "completed"
                }
                asyncio.create_task(manager.broadcast(msg))
            except Exception as ws_e:
                logger.warning(f"Failed to send WS completion: {ws_e}")

        else:
            running_jobs[job_id].update({
                'status': 'failed',
                'progress': 1.0,
                'error': result.error or 'Pipeline failed'
            })
            logger.error(f"Generation task {job_id} failed: {result.error}")
            
            # Notify WebSocket Fail
            try:
                from ..services.websocket_manager import manager
                msg = {
                    "type": "job_failed",
                    "job_id": job_id,
                    "error": result.error or 'Pipeline failed',
                    "status": "failed"
                }
                asyncio.create_task(manager.broadcast(msg))
            except Exception:
                pass

    except Exception as e:
        error_msg = f"Generation task failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        running_jobs[job_id].update({
            'status': 'failed',
            'progress': 1.0,
            'error': error_msg
        })

# Legacy endpoint for backward compatibility
@router.post("/{upload_id}")
async def generate_3d_legacy(upload_id: str):
    """
    Legacy endpoint for backward compatibility.
    """
    # For now, assume the image is in a standard location
    # In production, this would look up the uploaded file by ID
    image_path = f"uploads/{upload_id}.png"  # Placeholder

    if not os.path.exists(image_path):
        # Try alternative extensions
        for ext in ['.jpg', '.jpeg', '.png']:
            alt_path = f"uploads/{upload_id}{ext}"
            if os.path.exists(alt_path):
                image_path = alt_path
                break
        else:
            raise HTTPException(status_code=404, detail=f"Uploaded image not found for ID: {upload_id}")

    # Create request and delegate
    request = GenerateRequest(image_path=image_path)
    return await generate_3d(request, BackgroundTasks())
