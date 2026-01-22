import logging
import os
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter(prefix="/export", tags=["Export"])
logger = logging.getLogger(__name__)

@router.get("/{model_id}")
async def export_model(model_id: str, format: str = "ply"):
    """
    Exports the 3D model in the requested format (ply, glb, obj).
    """
    try:
        format = format.lower()
        if format not in ["ply", "glb", "obj"]:
             raise HTTPException(status_code=400, detail="Unsupported format. Use: ply, glb, obj")

        # Locate Model
        from ..core.config import settings
        model_dir = settings.PROJECT_ROOT / "assets" / "outputs" / model_id
        
        # Allow exporting refined vs original
        # Prioritize refined if available? No, user probably wants current state.
        # For simplicity, we look for 'reconstructed.ply' or '*.ply'
        # Ideally, frontend passes specific filename.
        input_ply = model_dir / "reconstructed.ply"
        if not input_ply.exists():
            refined_files = list(model_dir.glob("*_optimized.ply"))
            if refined_files:
                input_ply = refined_files[0]
            else:
                 ply_files = list(model_dir.glob("*.ply"))
                 if ply_files:
                     input_ply = ply_files[0]
                 else:
                     raise HTTPException(status_code=404, detail="Model file not found")

        # Case 1: PLY (Direct Pass-through)
        if format == "ply":
            return FileResponse(
                path=input_ply, 
                filename=f"{model_id}.ply", 
                media_type="application/octet-stream"
            )

        # Case 2: Conversion (GLB/OBJ)
        output_file = input_ply.parent / f"export.{format}"
        
        # Check if already cached
        if output_file.exists():
             return FileResponse(output_file, filename=f"{model_id}.{format}")

        logger.info(f"Converting {input_ply} -> {format}...")

        try:
            import trimesh
            import numpy as np
            import open3d as o3d
            from plyfile import PlyData
        except ImportError as e:
            logger.error(f"Missing libraries for conversion: {e}")
            raise HTTPException(status_code=500, detail="Conversion libraries (trimesh, open3d, plyfile) missing on server")

        try:
            # 1. Load PLY Data manually to extract SH coefficients
            plydata = PlyData.read(str(input_ply))
            vertex_data = plydata['vertex']
            
            # Extract positions
            x = vertex_data['x']
            y = vertex_data['y']
            z = vertex_data['z']
            positions = np.stack([x, y, z], axis=-1)
            
            # Extract Colors (SH Coefficients to RGB)
            # GS uses f_dc_0, f_dc_1, f_dc_2 for DC (0th order SH)
            # RGB = 0.5 + 0.28209 * DC
            
            colors = None
            if 'f_dc_0' in vertex_data:
                sh_dc = np.stack([
                    vertex_data['f_dc_0'],
                    vertex_data['f_dc_1'],
                    vertex_data['f_dc_2']
                ], axis=-1)
                
                # Conversion constant C0 = 0.28209479177387814
                C0 = 0.28209479177387814
                target_colors = 0.5 + C0 * sh_dc
                
                # Clamp to [0, 1]
                target_colors = np.clip(target_colors, 0.0, 1.0)
                
                # Convert to uint8
                colors = (target_colors * 255).astype(np.uint8)
            elif 'red' in vertex_data:
                # Fallback to standard PLY colors if available
                colors = np.stack([
                    vertex_data['red'],
                    vertex_data['green'],
                    vertex_data['blue']
                ], axis=-1).astype(np.uint8)
            else:
                # Default white
                colors = np.ones_like(positions) * 255
                colors = colors.astype(np.uint8)

            # 2. Reconstruct Mesh from Point Cloud (Poisson)
            # We use Open3D for robust Poisson reconstruction
            
            # Create Open3D PointCloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(positions)
            pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
            
            # Estimate normals if not present (crucial for Poisson)
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            # Orient normals consistent with tangent plane
            pcd.orient_normals_consistent_tangent_plane(100)
            
            # Poisson Reconstruction
            logger.info("Running Poisson Reconstruction...")
            with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
                mesh_o3d, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    pcd, depth=9, width=0, scale=1.1, linear_fit=False
                )
            
            # cleanup
            # Remove low density vertices (artifacts)
            vertices_to_remove = densities < np.quantile(densities, 0.05)
            mesh_o3d.remove_vertices_by_mask(vertices_to_remove)
            
            # Convert back to Trimesh for export (Open3D export is also fine, but Trimesh is robust for GLB)
            # Let's save to temp OBJ using Open3D and reload with Trimesh for final GLB conversion if needed
            # Or just export direct with Open3D if format supported.
            # Open3D supports .glb and .obj
            
            logger.info(f"Saving to {output_file}...")
            o3d.io.write_triangle_mesh(str(output_file), mesh_o3d)
            
            # Verify file exists and has size
            if not output_file.exists() or output_file.stat().st_size == 0:
                 raise RuntimeError("Export resulted in empty file")
                 
            return FileResponse(
                path=output_file, 
                filename=f"{model_id}.{format}",
                media_type="model/gltf-binary" if format == "glb" else "application/octet-stream"
            )

        except Exception as e:
            logger.error(f"Conversion failed: {e}")
            raise HTTPException(status_code=500, detail=f"Conversion failed: {e}")

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Export error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
