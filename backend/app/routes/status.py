"""
Status endpoint for real-time project monitoring.

Provides detailed status information for frontend to display
progress through the pipeline.
"""

from fastapi import APIRouter, HTTPException
from app.core.database import DatabaseManager
from app.core.supabase_client import get_supabase
from app.core.logger import logger

router = APIRouter(prefix="/status", tags=["Status"])


@router.get("/{project_id}")
async def get_project_status(project_id: str):
    """
    Get detailed status information for a project.
    
    Returns:
        project: Basic project info (status, current_step, etc.)
        multiview_count: Number of multi-view images generated
        depth_map_count: Number of depth maps created
        model_versions: List of model versions
        iterations: Refinement iteration metrics
        progress_percentage: Estimated completion percentage
    """
    try:
        # Get project basic info
        project = DatabaseManager.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        supabase = get_supabase()
        
        # Get multi-view generation count
        multiview_response = supabase.table("multiview_generation")\
            .select("id", count="exact")\
            .eq("project_id", project_id)\
            .execute()
        multiview_count = multiview_response.count or 0
        
        # Get depth map count
        depth_response = supabase.table("depth_maps")\
            .select("id", count="exact")\
            .eq("project_id", project_id)\
            .execute()
        depth_count = depth_response.count or 0
        
        # Get model versions
        models_response = supabase.table("gaussian_splat_models")\
            .select("version,model_file_url,num_splats,is_final")\
            .eq("project_id", project_id)\
            .order("version")\
            .execute()
        models = models_response.data
        
        # Get refinement iterations
        iterations_response = supabase.table("enhancement_iterations")\
            .select("iteration_number,overall_quality,psnr,ssim,converged")\
            .eq("project_id", project_id)\
            .order("iteration_number")\
            .execute()
        iterations = iterations_response.data
        
        # Calculate progress percentage
        progress = _calculate_progress(project["status"], multiview_count, depth_count, len(models))
        
        return {
            "project": {
                "id": project["id"],
                "status": project["status"],
                "current_step": project["current_step"],
                "error_message": project.get("error_message"),
                "original_image_url": project.get("original_image_url"),
                "processed_image_url": project.get("processed_image_url"),
                "final_model_url": project.get("final_model_url"),
                "total_processing_time": project.get("total_processing_time", 0),
                "created_at": project["created_at"]
            },
            "pipeline_progress": {
                "multiview_images": {
                    "completed": multiview_count,
                    "total": 16
                },
                "depth_maps": {
                    "completed": depth_count,
                    "total": 16
                },
                "model_versions": len(models),
                "refinement_iterations": len(iterations)
            },
            "models": models,
            "iterations": iterations,
            "progress_percentage": progress
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get status for project {project_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def _calculate_progress(status: str, multiview_count: int, depth_count: int, model_count: int) -> int:
    """
    Calculate estimated progress percentage based on pipeline status.
    
    Pipeline stages and their weights:
    - uploading/preprocessing: 0-10%
    - multiview_generating: 10-40%
    - depth_estimating: 40-60%
    - reconstructing: 60-70%
    - enhancing/refining: 70-95%
    - completed: 100%
    """
    if status == "uploading":
        return 5
    elif status == "preprocessing":
        return 10
    elif status == "multiview_generating":
        # 10% + (30% * completion ratio)
        return 10 + int(30 * (multiview_count / 16))
    elif status == "depth_estimating":
        # 40% + (20% * completion ratio)
        return 40 + int(20 * (depth_count / 16))
    elif status == "reconstructing":
        return 65
    elif status in ["enhancing", "refining"]:
        # 70% + (25% based on model versions)
        # Assume max 5 iterations
        return 70 + min(25, int(25 * (model_count / 5)))
    elif status == "completed":
        return 100
    elif status == "failed":
        return -1  # Special value for failed
    else:
        return 0


@router.get("/summary")
async def get_all_projects_summary():
    """
    Get a summary of all projects (useful for admin/monitoring).
    
    Returns:
        total_projects: Total number of projects
        by_status: Count of projects by status
        recent_projects: Last 10 projects created
    """
    try:
        supabase = get_supabase()
        
        # Get all projects
        response = supabase.table("projects")\
            .select("id,status,created_at,current_step")\
            .order("created_at", desc=True)\
            .execute()
        
        projects = response.data
        
        # Count by status
        status_counts = {}
        for project in projects:
            status = project["status"]
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Get recent projects (last 10)
        recent = projects[:10]
        
        return {
            "total_projects": len(projects),
            "by_status": status_counts,
            "recent_projects": recent
        }
        
    except Exception as e:
        logger.error(f"Failed to get projects summary: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
