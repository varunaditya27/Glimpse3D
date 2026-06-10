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

from ..core.config import settings
from ..core.logger import get_logger
from ..core.paths import is_within, new_job_id, safe_subdir
from ..core.repository import get_job_repo
from ..services.pipeline_manager import PipelineManager

logger = get_logger(__name__)
router = APIRouter(prefix="/generate", tags=["Generate"])

# Global pipeline manager instance.
# NOTE: run_pipeline is stateless per call (progress is delivered via an
# explicit per-job callback, not a shared callback list), so a single instance
# is safe to share across concurrent jobs.
pipeline_manager = PipelineManager()


def _resolve_input_image(image_path: str) -> Path:
    """Resolve a client-supplied image reference to a real path inside uploads.

    The client may pass either a bare stored filename or an absolute path it got
    back from /upload. Either way the resolved real path MUST live inside
    assets/uploads — otherwise the pipeline could be aimed at arbitrary files on
    the host (e.g. /etc/passwd). Raises HTTPException(400/404) on violation.
    """
    uploads_dir = (settings.PROJECT_ROOT / "assets" / "uploads").resolve()
    raw = (image_path or "").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="image_path is required.")

    candidate = Path(raw)
    if not candidate.is_absolute():
        # Treat a relative reference as a name under the uploads directory.
        candidate = uploads_dir / Path(raw.replace("\\", "/")).name

    if not is_within(uploads_dir, candidate):
        raise HTTPException(
            status_code=400,
            detail="image_path must reference a previously uploaded file.",
        )
    if not candidate.exists():
        raise HTTPException(status_code=404, detail="Uploaded image not found.")
    return candidate.resolve()

# Persistent job repository (backend chosen via settings.PERSISTENCE_BACKEND)
repo = get_job_repo()

class GenerateRequest(BaseModel):
    image_path: str
    output_dir: Optional[str] = None
    client_id: Optional[str] = None

class GenerateResponse(BaseModel):
    success: bool
    status: str
    model_url: Optional[str] = None
    error: Optional[str] = None
    job_id: Optional[str] = None

@router.post("/")
async def generate_3d(request: GenerateRequest, background_tasks: BackgroundTasks):
    """
    Triggers the coarse 3D reconstruction from the uploaded image.
    """
    try:
        # Validate + confine the input image to the uploads tree.
        image_path = str(_resolve_input_image(request.image_path))

        # Collision-free, restart-stable job id.
        job_id = new_job_id()

        # Output dir is ALWAYS server-controlled and confined to assets/outputs.
        # We never let the client choose an arbitrary write location.
        outputs_root = settings.PROJECT_ROOT / "assets" / "outputs"
        outputs_root.mkdir(parents=True, exist_ok=True)
        output_path = safe_subdir(outputs_root, job_id)
        output_path.mkdir(parents=True, exist_ok=True)
        output_dir = str(output_path)

        # Client that owns this job (for targeted WebSocket updates).
        # If None (older clients), we fall back to broadcast so nothing regresses.
        client_id = request.client_id

        # Store job status
        repo.create(job_id, image_path, output_dir)
        repo.update(job_id, status='starting', progress=0.0)

        # Run generation in background. Progress is delivered through a per-job
        # callback constructed inside the task (no shared mutable callback list).
        background_tasks.add_task(run_generation_task, job_id, image_path, output_dir, client_id)

        # Ensure cleanup loop is running
        ensure_cleanup_loop()

        return GenerateResponse(
            success=True,
            status="generation_started",
            job_id=job_id
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Generation request failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

import asyncio

# Timeout Manager
CLEANUP_task = None

async def job_timeout_monitor():
    """Periodically check for stale jobs."""
    logger.info("Starting Job Timeout Monitor...")
    while True:
        try:
            # Mark jobs still in starting/processing past the timeout as failed.
            # Note: We can't easily kill the asyncio task itself unless we stored the Task object
            # But marking it failed stops the frontend polling.
            reaped = repo.reap_stale(settings.PIPELINE_TIMEOUT)
            if reaped:
                logger.warning(f"Killed {reaped} stale job(s) (timeout: {settings.PIPELINE_TIMEOUT}s)")

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
    job = repo.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    return {
        'job_id': job['job_id'],
        'status': job['status'],
        'progress': job['progress'],
        'result': job.get('result'),
        'error': job.get('error'),
        'warnings': job.get('warnings', [])
    }

def _make_progress_callback(job_id: str, client_id: Optional[str]):
    """Build a per-job progress callback.

    Persists each pipeline state to this job's row and pushes a progress frame
    to the owning client (or broadcasts if the client_id is unknown). Because it
    closes over this job's id only, concurrent jobs never cross-write each
    other's rows or WebSocket streams — the bug the old shared callback list had.
    """
    from ..services.websocket_manager import manager

    def update_job_status(state):
        repo.update(
            job_id,
            status=state.stage.value,
            progress=state.progress,
            message=state.message,
            error=state.error,
            warnings=state.warnings,
        )
        msg = {
            "type": "progress_update",
            "job_id": job_id,
            "status": state.stage.value,
            "progress": state.progress,
            "message": state.message,
            "error": state.error,
            "warnings": state.warnings,
        }
        try:
            loop = asyncio.get_running_loop()
            if client_id:
                loop.create_task(manager.send_personal_message(msg, client_id))
            else:
                loop.create_task(manager.broadcast(msg))
        except RuntimeError:
            pass  # No running loop (e.g. sync context); skip live push.

    return update_job_status


async def run_generation_task(job_id: str, image_path: str, output_dir: str, client_id: Optional[str] = None):
    """Background task to run the generation pipeline."""
    try:
        logger.info(f"Starting generation task {job_id} for {image_path}")

        # Run FULL pipeline with a per-job progress callback.
        result = await pipeline_manager.run_pipeline(
            image_path, output_dir, progress_callback=_make_progress_callback(job_id, client_id)
        )

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

            repo.update(job_id, warnings=result.warnings)
            repo.mark_completed(job_id, {
                'model_url': model_url,
                'model_type': 'ply'
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
                if client_id:
                    asyncio.create_task(manager.send_personal_message(msg, client_id))
                else:
                    asyncio.create_task(manager.broadcast(msg))
            except Exception as ws_e:
                logger.warning(f"Failed to send WS completion: {ws_e}")

        else:
            repo.mark_failed(job_id, result.error or 'Pipeline failed', warnings=result.warnings)
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
                if client_id:
                    asyncio.create_task(manager.send_personal_message(msg, client_id))
                else:
                    asyncio.create_task(manager.broadcast(msg))
            except Exception:
                pass

    except Exception as e:
        error_msg = f"Generation task failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        repo.mark_failed(job_id, error_msg)

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
