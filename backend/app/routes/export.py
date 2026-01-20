"""
Handles final model export.

Responsibilities:
- Convert internal Gaussian Splat format to requested output format (.ply, .splat, .glb, .obj)
- Optimize file size if requested
- Provide download link
"""

import time
from fastapi import APIRouter, HTTPException, Query
from app.core.database import DatabaseManager
from app.core.storage import StorageManager
from app.core.logger import logger

router = APIRouter(prefix="/export", tags=["Export"])

@router.get("/{project_id}")
async def export_model(
    project_id: str,
    format: str = Query("ply", regex="^(ply|splat|glb|obj|usdz)$"),
    optimize: bool = False
):
    """
    Exports the 3D model in the requested format.
    
    Args:
        project_id: Project UUID
        format: Output format (ply, splat, glb, obj, usdz)
        optimize: Whether to optimize file size
        
    Returns:
        status: Export status
        download_url: URL to download the exported model
        format: Exported format
        file_size_mb: File size in megabytes
    """
    start_time = time.time()
    
    try:
        # Verify project exists
        project = DatabaseManager.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        if project["status"] not in ["refining", "completed"]:
            raise HTTPException(
                status_code=400,
                detail=f"Project is not ready for export. Current status: {project['status']}"
            )
        
        logger.info(f"Exporting project {project_id} to {format}")
        
        # TODO: Load the latest model version
        # from app.services.export_service import convert_model
        # supabase = get_supabase()
        # models = supabase.table("gaussian_splat_models").select("*").eq("project_id", project_id).order("version", desc=True).limit(1).execute()
        # latest_model_url = models.data[0]["model_file_url"]
        
        # TODO: Convert to requested format
        # exported_model = convert_model(latest_model_url, format, optimize)
        
        # For now, use placeholder
        exported_model = b"placeholder_model_data"
        
        # Upload exported model
        export_url = StorageManager.upload_exported_model(
            project_id, format, exported_model
        )
        
        # Calculate file size
        file_size_mb = len(exported_model) / (1024 * 1024)
        
        # Save export record
        optimization_level = "high" if optimize else "none"
        DatabaseManager.save_export(
            project_id,
            format,
            export_url,
            file_size_mb,
            optimization_level
        )
        
        # Update project if this is the first completion
        if project["status"] != "completed":
            elapsed_time = time.time() - start_time
            DatabaseManager.complete_project(
                project_id,
                export_url,
                project.get("total_processing_time", 0) + elapsed_time
            )
        
        logger.info(f"Export completed for project {project_id}")
        
        return {
            "status": "ready",
            "project_id": project_id,
            "download_url": export_url,
            "format": format,
            "file_size_mb": file_size_mb,
            "optimized": optimize
        }
        
    except Exception as e:
        logger.error(f"Export failed for project {project_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
