"""
Database helper functions for writing to Supabase tables.

Provides clean interfaces for inserting and updating records
throughout the Glimpse3D pipeline.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from app.core.supabase_client import get_supabase
from app.core.logger import logger


class DatabaseManager:
    """Handles all database operations for the pipeline."""
    
    # ==================== PROJECT MANAGEMENT ====================
    
    @staticmethod
    def create_project(user_session_id: Optional[str] = None) -> str:
        """
        Create a new project entry.
        
        Args:
            user_session_id: Optional session ID for tracking
            
        Returns:
            Project UUID
        """
        supabase = get_supabase()
        
        data = {
            "status": "uploading",
            "current_step": "Waiting for image upload",
        }
        
        if user_session_id:
            data["user_session_id"] = user_session_id
        
        response = supabase.table("projects").insert(data).execute()
        project_id = response.data[0]["id"]
        
        logger.info(f"Created project {project_id}")
        return project_id
    
    @staticmethod
    def update_project_status(
        project_id: str,
        status: str,
        current_step: Optional[str] = None,
        error_message: Optional[str] = None
    ):
        """Update project status and current step."""
        supabase = get_supabase()
        
        data = {"status": status}
        if current_step:
            data["current_step"] = current_step
        if error_message:
            data["error_message"] = error_message
        
        supabase.table("projects").update(data).eq("id", project_id).execute()
        logger.info(f"Project {project_id} status updated to {status}")
    
    @staticmethod
    def update_project_images(
        project_id: str,
        original_url: Optional[str] = None,
        processed_url: Optional[str] = None
    ):
        """Update project image URLs."""
        supabase = get_supabase()
        
        data = {}
        if original_url:
            data["original_image_url"] = original_url
        if processed_url:
            data["processed_image_url"] = processed_url
        
        supabase.table("projects").update(data).eq("id", project_id).execute()
    
    @staticmethod
    def complete_project(project_id: str, final_model_url: str, processing_time: float):
        """Mark project as completed."""
        supabase = get_supabase()
        
        data = {
            "status": "completed",
            "current_step": "Ready for export",
            "final_model_url": final_model_url,
            "total_processing_time": processing_time
        }
        
        supabase.table("projects").update(data).eq("id", project_id).execute()
        logger.info(f"Project {project_id} completed in {processing_time:.2f}s")
    
    @staticmethod
    def get_project(project_id: str) -> Optional[Dict]:
        """Retrieve project details."""
        supabase = get_supabase()
        response = supabase.table("projects").select("*").eq("id", project_id).execute()
        return response.data[0] if response.data else None
    
    # ==================== MULTIVIEW GENERATION ====================
    
    @staticmethod
    def save_multiview_images(
        project_id: str,
        views_data: List[Dict[str, Any]]
    ):
        """
        Save multiple multi-view images.
        
        Args:
            project_id: Project UUID
            views_data: List of dicts with keys: view_index, elevation, azimuth, image_url
        """
        supabase = get_supabase()
        
        records = []
        for view in views_data:
            records.append({
                "project_id": project_id,
                "view_index": view["view_index"],
                "elevation": view.get("elevation"),
                "azimuth": view.get("azimuth"),
                "image_url": view["image_url"]
            })
        
        supabase.table("multiview_generation").insert(records).execute()
        logger.info(f"Saved {len(records)} multi-view images for project {project_id}")
    
    @staticmethod
    def save_single_multiview(
        project_id: str,
        view_index: int,
        image_url: str,
        elevation: Optional[float] = None,
        azimuth: Optional[float] = None
    ):
        """Save a single multi-view image."""
        supabase = get_supabase()
        
        data = {
            "project_id": project_id,
            "view_index": view_index,
            "image_url": image_url,
            "elevation": elevation,
            "azimuth": azimuth
        }
        
        supabase.table("multiview_generation").insert(data).execute()
    
    # ==================== DEPTH MAPS ====================
    
    @staticmethod
    def save_depth_maps(
        project_id: str,
        depth_data: List[Dict[str, Any]]
    ):
        """
        Save multiple depth maps.
        
        Args:
            project_id: Project UUID
            depth_data: List of dicts with depth map information
        """
        supabase = get_supabase()
        
        records = []
        for depth in depth_data:
            records.append({
                "project_id": project_id,
                "view_index": depth["view_index"],
                "depth_map_url": depth["depth_map_url"],
                "depth_heatmap_url": depth.get("depth_heatmap_url"),
                "min_depth": depth.get("min_depth"),
                "max_depth": depth.get("max_depth"),
                "mean_depth": depth.get("mean_depth"),
                "confidence_score": depth.get("confidence_score")
            })
        
        supabase.table("depth_maps").insert(records).execute()
        logger.info(f"Saved {len(records)} depth maps for project {project_id}")
    
    @staticmethod
    def save_single_depth_map(
        project_id: str,
        view_index: int,
        depth_map_url: str,
        depth_heatmap_url: Optional[str] = None,
        min_depth: Optional[float] = None,
        max_depth: Optional[float] = None,
        mean_depth: Optional[float] = None,
        confidence_score: Optional[float] = None
    ):
        """Save a single depth map."""
        supabase = get_supabase()
        
        data = {
            "project_id": project_id,
            "view_index": view_index,
            "depth_map_url": depth_map_url,
            "depth_heatmap_url": depth_heatmap_url,
            "min_depth": min_depth,
            "max_depth": max_depth,
            "mean_depth": mean_depth,
            "confidence_score": confidence_score
        }
        
        supabase.table("depth_maps").insert(data).execute()
    
    # ==================== GAUSSIAN SPLAT MODELS ====================
    
    @staticmethod
    def save_gaussian_model(
        project_id: str,
        version: int,
        model_url: str,
        num_splats: Optional[int] = None,
        file_size_mb: Optional[float] = None,
        is_final: bool = False
    ) -> str:
        """
        Save a Gaussian Splat model version.
        
        Returns:
            Model UUID
        """
        supabase = get_supabase()
        
        data = {
            "project_id": project_id,
            "version": version,
            "model_file_url": model_url,
            "num_splats": num_splats,
            "file_size_mb": file_size_mb,
            "is_final": is_final
        }
        
        response = supabase.table("gaussian_splat_models").insert(data).execute()
        model_id = response.data[0]["id"]
        
        logger.info(f"Saved model v{version} for project {project_id}")
        return model_id
    
    # ==================== ENHANCEMENT ITERATIONS ====================
    
    @staticmethod
    def create_enhancement_iteration(
        project_id: str,
        iteration_number: int,
        learning_rate: Optional[float] = None
    ) -> str:
        """
        Create a new enhancement iteration entry.
        
        Returns:
            Iteration UUID
        """
        supabase = get_supabase()
        
        data = {
            "project_id": project_id,
            "iteration_number": iteration_number,
            "learning_rate": learning_rate
        }
        
        response = supabase.table("enhancement_iterations").insert(data).execute()
        iteration_id = response.data[0]["id"]
        
        logger.info(f"Created iteration {iteration_number} for project {project_id}")
        return iteration_id
    
    @staticmethod
    def update_iteration_metrics(
        iteration_id: str,
        views_processed: Optional[int] = None,
        avg_depth_consistency: Optional[float] = None,
        avg_feature_similarity: Optional[float] = None,
        psnr: Optional[float] = None,
        ssim: Optional[float] = None,
        lpips: Optional[float] = None,
        overall_quality: Optional[float] = None,
        converged: Optional[bool] = None,
        processing_time: Optional[float] = None
    ):
        """Update metrics for an enhancement iteration."""
        supabase = get_supabase()
        
        data = {}
        if views_processed is not None:
            data["views_processed"] = views_processed
        if avg_depth_consistency is not None:
            data["avg_depth_consistency"] = avg_depth_consistency
        if avg_feature_similarity is not None:
            data["avg_feature_similarity"] = avg_feature_similarity
        if psnr is not None:
            data["psnr"] = psnr
        if ssim is not None:
            data["ssim"] = ssim
        if lpips is not None:
            data["lpips"] = lpips
        if overall_quality is not None:
            data["overall_quality"] = overall_quality
        if converged is not None:
            data["converged"] = converged
        if processing_time is not None:
            data["processing_time"] = processing_time
        
        supabase.table("enhancement_iterations").update(data).eq("id", iteration_id).execute()
    
    # ==================== ENHANCED VIEWS ====================
    
    @staticmethod
    def save_enhanced_view(
        iteration_id: str,
        view_index: int,
        enhanced_image_url: str,
        rendered_image_url: Optional[str] = None,
        prompt_used: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        controlnet_scale: Optional[float] = None,
        guidance_scale: Optional[float] = None,
        num_inference_steps: Optional[int] = None
    ):
        """Save an enhanced view for an iteration."""
        supabase = get_supabase()
        
        data = {
            "iteration_id": iteration_id,
            "view_index": view_index,
            "enhanced_image_url": enhanced_image_url,
            "rendered_image_url": rendered_image_url,
            "prompt_used": prompt_used,
            "negative_prompt": negative_prompt,
            "controlnet_scale": controlnet_scale,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps
        }
        
        supabase.table("enhanced_views").insert(data).execute()
    
    # ==================== REFINEMENT METRICS ====================
    
    @staticmethod
    def save_refinement_metrics(
        iteration_id: str,
        metrics: Dict[str, float],
        view_index: Optional[int] = None,
        improvement_over_baseline: Optional[Dict[str, float]] = None
    ):
        """
        Save detailed refinement metrics.
        
        Args:
            iteration_id: Iteration UUID
            metrics: Dict of metric_name -> metric_value
            view_index: If metrics are per-view
            improvement_over_baseline: Dict of metric_name -> improvement value
        """
        supabase = get_supabase()
        
        records = []
        for metric_name, metric_value in metrics.items():
            improvement = None
            if improvement_over_baseline and metric_name in improvement_over_baseline:
                improvement = improvement_over_baseline[metric_name]
            
            records.append({
                "iteration_id": iteration_id,
                "metric_name": metric_name,
                "metric_value": float(metric_value),
                "improvement_over_baseline": improvement,
                "view_index": view_index
            })
        
        supabase.table("refinement_metrics").insert(records).execute()
    
    # ==================== EXPORT HISTORY ====================
    
    @staticmethod
    def save_export(
        project_id: str,
        format: str,
        file_url: str,
        file_size_mb: Optional[float] = None,
        optimization_level: Optional[str] = None
    ) -> str:
        """
        Save an export record.
        
        Returns:
            Export UUID
        """
        supabase = get_supabase()
        
        data = {
            "project_id": project_id,
            "format": format,
            "file_url": file_url,
            "file_size_mb": file_size_mb,
            "optimization_level": optimization_level
        }
        
        response = supabase.table("export_history").insert(data).execute()
        export_id = response.data[0]["id"]
        
        logger.info(f"Saved export record for project {project_id} (format: {format})")
        return export_id
