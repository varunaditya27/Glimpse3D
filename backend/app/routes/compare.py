"""
Compare endpoint for side-by-side model visualization.

Provides functionality to:
- List all completed projects for selection
- Fetch dual model data for comparison
"""

from fastapi import APIRouter, HTTPException
from app.core.database import DatabaseManager
from app.core.supabase_client import get_supabase
from app.core.logger import logger
from typing import Optional

router = APIRouter(prefix="/compare", tags=["Compare"])


@router.get("/projects")
async def get_projects_for_comparison(limit: Optional[int] = 50, status: Optional[str] = "completed"):
    """
    Get a list of all projects available for comparison.
    
    Args:
        limit: Maximum number of projects to return (default: 50)
        status: Filter by project status (default: 'completed')
    
    Returns:
        List of projects with:
        - id, original_image_url, created_at
        - latest_model_url, final_model_url
        - project status and metrics
    """
    try:
        projects = DatabaseManager.get_projects_for_comparison(limit=limit, status=status)
        
        return {
            "total": len(projects),
            "projects": projects
        }
        
    except Exception as e:
        logger.error(f"Failed to get projects for comparison: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{project_id_1}/{project_id_2}")
async def get_dual_model_data(project_id_1: str, project_id_2: str):
    """
    Fetch data for two projects to enable side-by-side comparison.
    
    Args:
        project_id_1: First project UUID
        project_id_2: Second project UUID
    
    Returns:
        project_a: Full data for first project
        project_b: Full data for second project
        Each includes:
            - Basic project info
            - Latest model URL and metadata
            - Refinement metrics
            - Multi-view counts
    """
    try:
        # Fetch both projects
        project_a_data = DatabaseManager.get_project_comparison_data(project_id_1)
        project_b_data = DatabaseManager.get_project_comparison_data(project_id_2)
        
        if not project_a_data:
            raise HTTPException(status_code=404, detail=f"Project {project_id_1} not found")
        if not project_b_data:
            raise HTTPException(status_code=404, detail=f"Project {project_id_2} not found")
        
        return {
            "project_a": project_a_data,
            "project_b": project_b_data,
            "comparison": {
                "same_status": project_a_data["status"] == project_b_data["status"],
                "time_difference_seconds": abs(
                    project_a_data.get("total_processing_time", 0) - 
                    project_b_data.get("total_processing_time", 0)
                )
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get dual model data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/project/{project_id}/models")
async def get_project_model_versions(project_id: str):
    """
    Get all model versions for a project (useful for version comparison).
    
    Args:
        project_id: Project UUID
    
    Returns:
        List of all model versions with URLs and metadata
    """
    try:
        supabase = get_supabase()
        
        # Get all models for this project
        models_response = supabase.table("gaussian_splat_models")\
            .select("*")\
            .eq("project_id", project_id)\
            .order("version")\
            .execute()
        
        models = models_response.data
        
        if not models:
            raise HTTPException(status_code=404, detail=f"No models found for project {project_id}")
        
        return {
            "project_id": project_id,
            "total_versions": len(models),
            "models": models
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model versions for project {project_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
