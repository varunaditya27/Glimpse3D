import argparse
import logging
import sys
import os
import gc
from pathlib import Path
from typing import Optional, Any, Dict

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('reconstruct.log')
    ]
)
logger = logging.getLogger("Glimpse3D-Reconstruct")

# Add TripoSR to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../TripoSR")))

import torch
# Monkeypatch for transformers/torch compatibility issue
if not hasattr(torch.utils, "_pytree"):
    import torch.utils._pytree
if not hasattr(torch.utils._pytree, "register_pytree_node"):
    def register_pytree_node(cls, flatten_fn, unflatten_fn, serialized_type_name=None):
        pass # No-op
    torch.utils._pytree.register_pytree_node = register_pytree_node

try:
    from tsr.system import TSR
except ImportError:
    TSR = None

from PIL import Image
import numpy as np

try:
    from rembg import remove
except ImportError:
    remove = None

try:
    from plyfile import PlyData, PlyElement
except ImportError:
    PlyData = None
    logger.warning("plyfile not found. PLY export will fail.")

class SimpleMesh:
    """Fallback mesh class when trimesh is missing."""
    def __init__(self, vertices, faces):
        self.vertices = vertices
        self.faces = faces

    def export(self, path):
        pass

class TripoInference:
    """
    Singleton class for TripoSR Inference.
    Keeps model in memory to avoid reloading on every request.
    """
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TripoInference, cls).__new__(cls)
        return cls._instance

    def load_model(self):
        """Load TripoSR model if not already loaded."""
        if self._model is not None:
            return self._model

        if TSR is None:
            logger.error("TripoSR (tsr) package not found.")
            raise ImportError("Missing dependency: tsr")

        logger.info("Loading TripoSR model...")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        try:
            model = TSR.from_pretrained(
                "stabilityai/TripoSR",
                config_name="config.yaml",
                weight_name="model.ckpt",
            )
            model.renderer.set_chunk_size(131072)
            model.to(device)
            self._model = model
            logger.info(f"Model loaded successfully on {device}")
            return self._model
        except Exception as e:
            logger.error(f"Failed to load TripoSR model: {e}")
            raise

    def preprocess_image(self, image_path: str):
        """Load and preprocess image."""
        logger.info(f"Preprocessing image: {image_path}")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGBA")

        if remove is not None:
            image = remove(image)
        else:
            logger.warning("rembg not installed. Skipping background removal.")

        image_np = np.array(image).astype(np.float32) / 255.0
        if image_np.shape[2] == 4:
            image_np = image_np[:, :, :3] * image_np[:, :, 3:4] + (1 - image_np[:, :, 3:4]) * 0.5
            image = Image.fromarray((image_np * 255.0).astype(np.uint8))
        else:
            image = image.convert("RGB")

        image = image.resize((512, 512), Image.BILINEAR)
        return image

    def run_inference(self, processed_image):
        """Run inference on processed image."""
        model = self.load_model()
        logger.info("Running inference...")

        if model is not None:
            try:
                device = next(model.parameters()).device
                with torch.no_grad():
                    scene_codes = model([processed_image], device=device)
                    meshes = model.extract_mesh(scene_codes, True, resolution=256)
                    return meshes[0]
            except Exception as e:
                logger.error(f"Inference failed (fallback): {e}")

        # Fallback logic
        logger.warning("Using Dummy Mesh (Sphere/Cube) for testing.")
        verts = []
        for x in [-0.5, 0.5]:
            for y in [-0.5, 0.5]:
                for z in [-0.5, 0.5]:
                    verts.append([x, y, z])
        return SimpleMesh(np.array(verts, dtype=np.float32), np.array([[0, 1, 2]], dtype=np.int32))

    def export_ply(self, mesh, output_path: str):
        """Sample points and export to PLY."""
        logger.info(f"Exporting to {output_path}...")
        
        if hasattr(mesh, "sample"):
            points = mesh.sample(100000)
        else:
            points = mesh.vertices
            if len(points) < 100:
                points = np.random.uniform(-0.5, 0.5, (1000, 3)).astype(np.float32)

        num_points = len(points)
        
        # Initialize Attributes
        xyz = points
        features_dc = np.zeros((num_points, 3), dtype=np.float32)
        features_rest = np.zeros((num_points, 45), dtype=np.float32)
        opacities = np.ones((num_points, 1), dtype=np.float32) * 2.0
        scales = np.ones((num_points, 3), dtype=np.float32) * np.log(0.01)
        rots = np.repeat(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), num_points, axis=0)

        # Construct Structured Array
        dtype = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
            ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
        ]
        for i in range(45): dtype.append((f'f_rest_{i}', 'f4'))
        dtype.append(('opacity', 'f4'))
        dtype.append(('scale_0', 'f4'))
        dtype.append(('scale_1', 'f4'))
        dtype.append(('scale_2', 'f4'))
        dtype.append(('rot_0', 'f4'))
        dtype.append(('rot_1', 'f4'))
        dtype.append(('rot_2', 'f4'))
        dtype.append(('rot_3', 'f4'))

        elements = np.empty(num_points, dtype=dtype)
        elements['x'] = xyz[:, 0]
        elements['y'] = xyz[:, 1]
        elements['z'] = xyz[:, 2]
        elements['nx'] = 0; elements['ny'] = 0; elements['nz'] = 0
        elements['red'] = 200; elements['green'] = 200; elements['blue'] = 200
        elements['f_dc_0'] = features_dc[:, 0]
        elements['f_dc_1'] = features_dc[:, 1]
        elements['f_dc_2'] = features_dc[:, 2]
        for i in range(45): elements[f'f_rest_{i}'] = features_rest[:, i]
        elements['opacity'] = opacities[:, 0]
        elements['scale_0'] = scales[:, 0]; elements['scale_1'] = scales[:, 1]; elements['scale_2'] = scales[:, 2]
        elements['rot_0'] = rots[:, 0]; elements['rot_1'] = rots[:, 1]; elements['rot_2'] = rots[:, 2]; elements['rot_3'] = rots[:, 3]

        if PlyData is not None:
            el = PlyElement.describe(elements, 'vertex')
            PlyData([el]).write(output_path)
            logger.info(f"PLY saved to {output_path}")
        else:
            logger.error("Cannot save PLY: plyfile module missing.")

    def run(self, image_path: str, output_path: str):
        """Main entry point for pipeline."""
        try:
            self.load_model()
            image = self.preprocess_image(image_path)
            mesh = self.run_inference(image)
            self.export_ply(mesh, output_path)
            
            # VRAM Cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            return True
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return False

# Compatibility wrappers for existing scripts
def load_triposr_model(): return TripoInference().load_model()
def preprocess_image(path): return TripoInference().preprocess_image(path)
def run_inference(model, img): return TripoInference().run_inference(img)
def sample_and_export(mesh, path): return TripoInference().export_ply(mesh, path)

def main():
    parser = argparse.ArgumentParser(description="Glimpse3D Reconstruction")
    parser.add_argument("image_path", type=str, help="Path to input image")
    parser.add_argument("--output", type=str, default="output.ply", help="Path to save output PLY")
    args = parser.parse_args()

    inference = TripoInference()
    success = inference.run(args.image_path, args.output)
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
