"""
ai_modules/colab_utils.py

Utility functions for running Glimpse3D pipeline on Google Colab.

This module provides simplified interfaces and helper functions that work
seamlessly in the Colab environment.
"""

import os
import sys
from pathlib import Path
from typing import Optional, List, Union, Tuple
import numpy as np

# Check environment
IN_COLAB = 'google.colab' in sys.modules


def setup_colab_environment():
    """
    Setup Google Colab environment for Glimpse3D.
    
    Configures paths, GPU settings, and installs dependencies if needed.
    """
    if not IN_COLAB:
        print("Not running in Colab. Environment setup skipped.")
        return
    
    import torch
    
    # GPU optimization
    if torch.cuda.is_available():
        # Enable TF32 for faster matrix operations
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        
        # Memory optimization
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
        
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ GPU configured: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("⚠️ No GPU detected. Performance will be limited.")


def check_dependencies() -> dict:
    """
    Check which Glimpse3D dependencies are available.
    
    Returns:
        Dictionary with module names and their availability status.
    """
    deps = {}
    
    # Core
    try:
        import torch
        deps['torch'] = f"✅ {torch.__version__}"
    except ImportError:
        deps['torch'] = "❌ Not installed"
    
    # TripoSR
    try:
        from tsr.system import TSR
        deps['triposr'] = "✅ Available"
    except ImportError:
        deps['triposr'] = "❌ Clone github.com/VAST-AI-Research/TripoSR"
    
    # gsplat
    try:
        import gsplat
        deps['gsplat'] = f"✅ {gsplat.__version__}"
    except ImportError:
        deps['gsplat'] = "❌ pip install gsplat"
    
    # Diffusers
    try:
        import diffusers
        deps['diffusers'] = f"✅ {diffusers.__version__}"
    except ImportError:
        deps['diffusers'] = "❌ pip install diffusers"
    
    # SyncDreamer deps
    try:
        import omegaconf
        import pytorch_lightning
        deps['syncdreamer_deps'] = "✅ Available"
    except ImportError:
        deps['syncdreamer_deps'] = "❌ pip install omegaconf pytorch-lightning"
    
    # Image processing
    try:
        import rembg
        deps['rembg'] = "✅ Available"
    except ImportError:
        deps['rembg'] = "❌ pip install rembg[gpu]"
    
    try:
        from plyfile import PlyData
        deps['plyfile'] = "✅ Available"
    except ImportError:
        deps['plyfile'] = "❌ pip install plyfile"
    
    return deps


def print_dependencies():
    """Print dependency status in a formatted way."""
    deps = check_dependencies()
    print("\n" + "="*50)
    print("📦 Glimpse3D Dependencies")
    print("="*50)
    for name, status in deps.items():
        print(f"  {name}: {status}")
    print("="*50 + "\n")


def clear_gpu_memory():
    """Clear GPU memory cache."""
    import gc
    import torch
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"🧹 GPU memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")


def download_file(url: str, output_path, use_aria2: bool = True) -> bool:
    """
    Download a file with progress display.
    
    Args:
        url: Download URL
        output_path: Where to save the file
        use_aria2: Use aria2c for faster downloads (Colab only)
    
    Returns:
        True if download succeeded
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.exists():
        print(f"✅ File exists: {output_path}")
        return True
    
    if IN_COLAB and use_aria2:
        import subprocess
        # Install aria2 if not present
        subprocess.run(['apt', '-y', 'install', '-qq', 'aria2'], capture_output=True)
        
        result = subprocess.run([
            'aria2c',
            '--console-log-level=error',
            '-c', '-x', '16', '-s', '16', '-k', '1M',
            url,
            '-d', str(output_path.parent),
            '-o', output_path.name
        ])
        return result.returncode == 0
    else:
        # Use urllib
        import urllib.request
        try:
            urllib.request.urlretrieve(url, str(output_path))
            return True
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return False


class GaussianSplatIO:
    """Utilities for reading/writing Gaussian Splat PLY files."""
    
    @staticmethod
    def load(path: str) -> dict:
        """
        Load Gaussian Splat PLY file.
        
        Returns:
            Dictionary with xyz, f_dc, f_rest, opacity, scales, rotations
        """
        from plyfile import PlyData
        import torch
        
        plydata = PlyData.read(path)
        vertex = plydata['vertex']
        
        xyz = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=-1)
        f_dc = np.stack([vertex['f_dc_0'], vertex['f_dc_1'], vertex['f_dc_2']], axis=-1)
        
        # Load f_rest coefficients
        f_rest_names = [f'f_rest_{i}' for i in range(45)]
        available_names = [n for n in f_rest_names if n in vertex.data.dtype.names]
        if available_names:
            f_rest = np.stack([vertex[n] for n in available_names], axis=-1)
        else:
            f_rest = np.zeros((len(xyz), 45), dtype=np.float32)
        
        opacity = vertex['opacity']
        scales = np.stack([vertex['scale_0'], vertex['scale_1'], vertex['scale_2']], axis=-1)
        rotations = np.stack([vertex['rot_0'], vertex['rot_1'], vertex['rot_2'], vertex['rot_3']], axis=-1)
        
        return {
            'xyz': torch.tensor(xyz, dtype=torch.float32),
            'f_dc': torch.tensor(f_dc, dtype=torch.float32),
            'f_rest': torch.tensor(f_rest, dtype=torch.float32),
            'opacity': torch.tensor(opacity, dtype=torch.float32),
            'scales': torch.tensor(scales, dtype=torch.float32),
            'rotations': torch.tensor(rotations, dtype=torch.float32),
        }
    
    @staticmethod
    def save(gaussians: dict, path: str):
        """
        Save Gaussian Splat data to PLY file.
        
        Args:
            gaussians: Dictionary with xyz, f_dc, f_rest, opacity, scales, rotations
            path: Output file path
        """
        from plyfile import PlyElement, PlyData
        
        # Convert to numpy if tensor
        def to_numpy(x):
            if hasattr(x, 'cpu'):
                return x.cpu().numpy()
            return np.array(x)
        
        xyz = to_numpy(gaussians['xyz'])
        f_dc = to_numpy(gaussians['f_dc'])
        f_rest = to_numpy(gaussians['f_rest'])
        opacity = to_numpy(gaussians['opacity'])
        scales = to_numpy(gaussians['scales'])
        rotations = to_numpy(gaussians['rotations'])
        
        num_points = len(xyz)
        
        # Build dtype
        dtype_list = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
        ]
        for i in range(f_rest.shape[1] if len(f_rest.shape) > 1 else 45):
            dtype_list.append((f'f_rest_{i}', 'f4'))
        dtype_list.extend([
            ('opacity', 'f4'),
            ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
            ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
        ])
        
        elements = np.zeros(num_points, dtype=dtype_list)
        elements['x'] = xyz[:, 0]
        elements['y'] = xyz[:, 1]
        elements['z'] = xyz[:, 2]
        elements['f_dc_0'] = f_dc[:, 0]
        elements['f_dc_1'] = f_dc[:, 1]
        elements['f_dc_2'] = f_dc[:, 2]
        
        for i in range(f_rest.shape[1] if len(f_rest.shape) > 1 else 0):
            elements[f'f_rest_{i}'] = f_rest[:, i]
        
        elements['opacity'] = opacity.flatten()
        elements['scale_0'] = scales[:, 0]
        elements['scale_1'] = scales[:, 1]
        elements['scale_2'] = scales[:, 2]
        elements['rot_0'] = rotations[:, 0]
        elements['rot_1'] = rotations[:, 1]
        elements['rot_2'] = rotations[:, 2]
        elements['rot_3'] = rotations[:, 3]
        
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        print(f"✅ Saved: {path}")


class CameraUtils:
    """Camera utilities for SyncDreamer-compatible views."""
    
    # SyncDreamer camera configuration
    ELEVATIONS = [30.0] * 8 + [-20.0] * 8
    AZIMUTHS = [i * 45.0 for i in range(8)] * 2
    
    @staticmethod
    def create_pose(elevation_deg: float, azimuth_deg: float, radius: float = 2.0) -> np.ndarray:
        """
        Create world-to-camera matrix.
        
        Args:
            elevation_deg: Elevation angle in degrees
            azimuth_deg: Azimuth angle in degrees
            radius: Distance from origin
        
        Returns:
            4x4 world-to-camera transformation matrix
        """
        import math
        
        elev = math.radians(elevation_deg)
        azim = math.radians(azimuth_deg)
        
        x = radius * math.cos(elev) * math.cos(azim)
        y = radius * math.cos(elev) * math.sin(azim)
        z = radius * math.sin(elev)
        
        cam_pos = np.array([x, y, z])
        look_at = np.array([0, 0, 0])
        up = np.array([0, 0, 1])
        
        forward = look_at - cam_pos
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        
        up_new = np.cross(right, forward)
        
        w2c = np.eye(4)
        w2c[:3, 0] = right
        w2c[:3, 1] = up_new
        w2c[:3, 2] = -forward
        w2c[:3, 3] = -w2c[:3, :3] @ cam_pos
        
        return w2c
    
    @staticmethod
    def get_syncdreamer_poses(radius: float = 2.0) -> List[np.ndarray]:
        """Get all 16 camera poses matching SyncDreamer configuration."""
        poses = []
        for elev, azim in zip(CameraUtils.ELEVATIONS, CameraUtils.AZIMUTHS):
            poses.append(CameraUtils.create_pose(elev, azim, radius))
        return poses
    
    @staticmethod
    def get_projection_matrix(fov_deg: float = 60, aspect: float = 1.0) -> np.ndarray:
        """Create projection matrix."""
        import math
        
        fov_rad = math.radians(fov_deg)
        f = 1.0 / math.tan(fov_rad / 2)
        
        proj = np.zeros((4, 4))
        proj[0, 0] = f / aspect
        proj[1, 1] = f
        proj[2, 2] = -1.01  # (far + near) / (near - far) for near=0.1, far=100
        proj[2, 3] = -0.2   # 2 * far * near / (near - far)
        proj[3, 2] = -1
        
        return proj


def mesh_to_gaussian_ply(mesh, output_path: str, num_samples: int = 100000):
    """
    Convert a trimesh mesh to Gaussian Splat PLY format.
    
    Args:
        mesh: trimesh mesh object
        output_path: Path to save output PLY
        num_samples: Number of points to sample
    """
    print(f"🔄 Sampling {num_samples:,} points from mesh...")
    
    # Sample points from mesh
    if hasattr(mesh, 'sample'):
        points, face_indices = mesh.sample(num_samples, return_index=True)
    else:
        # Fallback for non-trimesh objects
        points = np.array(mesh.vertices)
        face_indices = None
    
    # Get colors if available
    if hasattr(mesh, 'visual') and mesh.visual.vertex_colors is not None:
        if face_indices is not None:
            face_vertices = mesh.faces[face_indices]
            vertex_colors = mesh.visual.vertex_colors[:, :3] / 255.0
            colors = vertex_colors[face_vertices].mean(axis=1)
        else:
            colors = mesh.visual.vertex_colors[:, :3] / 255.0
    else:
        colors = np.ones((len(points), 3)) * 0.5
    
    num_points = len(points)
    
    # Convert colors to SH DC format
    C0 = 0.28209479177387814
    features_dc = ((colors - 0.5) / C0).astype(np.float32)
    
    gaussians = {
        'xyz': points.astype(np.float32),
        'f_dc': features_dc,
        'f_rest': np.zeros((num_points, 45), dtype=np.float32),
        'opacity': np.ones((num_points,), dtype=np.float32) * 2.2,  # Pre-sigmoid
        'scales': np.ones((num_points, 3), dtype=np.float32) * (-4.6),  # Pre-exp
        'rotations': np.tile([1.0, 0.0, 0.0, 0.0], (num_points, 1)).astype(np.float32),
    }
    
    GaussianSplatIO.save(gaussians, output_path)


# Convenience exports
__all__ = [
    'IN_COLAB',
    'setup_colab_environment',
    'check_dependencies',
    'print_dependencies',
    'clear_gpu_memory',
    'download_file',
    'GaussianSplatIO',
    'CameraUtils',
    'mesh_to_gaussian_ply',
]
