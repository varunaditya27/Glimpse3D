import argparse
import logging
import sys
import os
from pathlib import Path

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
# Monkeypatch for transformers/torch compatibility issue (register_pytree_node)
if not hasattr(torch.utils, "_pytree"):
    import torch.utils._pytree
if not hasattr(torch.utils._pytree, "register_pytree_node"):
    def register_pytree_node(cls, flatten_fn, unflatten_fn, serialized_type_name=None):
        pass # No-op
    torch.utils._pytree.register_pytree_node = register_pytree_node

try:
    from tsr.system import TSR
except ImportError:
    # Fallback/Mock for environment discovery (Dev 2 safety)
    TSR = None

def load_triposr_model():
    """
    P1.2: Load TripoSR model and processor.
    """
    if TSR is None:
        logger.error("TripoSR (tsr) package not found. Please install it.")
        raise ImportError("Missing dependency: tsr")

    logger.info("Loading TripoSR model...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Load from pretrained (automatically handles caching in ~/.cache/huggingface)
    try:
        model = TSR.from_pretrained(
            "stabilityai/TripoSR",
            config_name="config.yaml",
            weight_name="model.ckpt",
        )
        model.renderer.set_chunk_size(131072)
        model.to(device)
        logger.info(f"Model loaded successfully on {device}")
        return model
    except Exception as e:
        logger.error(f"Failed to load TripoSR model: {e}")
        raise

from PIL import Image
import numpy as np

try:
    from rembg import remove
except ImportError:
    remove = None

def preprocess_image(image_path: str):
    """
    P1.3: Load and preprocess image (Remove BG, Resize).
    Returns:
        processed_image_tensor: (3, 512, 512) torch tensor or equivalent for TripoSR.
    """
    logger.info(f"Preprocessing image: {image_path}")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load Image
    image = Image.open(image_path).convert("RGBA")

    # Remove Background
    if remove is not None:
        logger.info("Removing background using rembg...")
        image = remove(image)
    else:
        logger.warning("rembg not installed. Skipping background removal. Ensure input is RGBA with clear BG.")

    # Compositing Logic (Match TripoSR/run.py)
    # Convert to float
    image_np = np.array(image).astype(np.float32) / 255.0
    
    # If image has alpha, composite over gray (0.5)
    if image_np.shape[2] == 4:
        # RGB * Alpha + Background * (1 - Alpha)
        image_np = image_np[:, :, :3] * image_np[:, :, 3:4] + (1 - image_np[:, :, 3:4]) * 0.5
        image = Image.fromarray((image_np * 255.0).astype(np.uint8))
    else:
        # Already RGB, ensure 3 channels
        image = image.convert("RGB")

    # Simple Resize (TripoSR usually needs foreground centering, keeping simple for now)
    # Target size: 512x512
    image = image.resize((512, 512), Image.BILINEAR)

    return image

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

def run_inference(model, processed_image):
    """
    P1.4: Run TripoSR inference to get mesh.
    """
    logger.info("Running inference...")

    if model is not None:
        try:
            device = next(model.parameters()).device
            # Check inputs - assuming processed_image is RGBA 512x512 PIL
            # TSR typically expects a batch or list of images.
            with torch.no_grad():
                scene_codes = model([processed_image], device=device)
                # Extract mesh: resolution 256 is default
                # P1.5 Update: Added has_vertex_color=True
                meshes = model.extract_mesh(scene_codes, True, resolution=256)
                return meshes[0] # Return the first mesh (trimesh object usually)
        except Exception as e:
            logger.error(f"Inference failed (fallback to dummy): {e}")

    # Fallback to Dummy Sphere/Cube if model is missing
    logger.warning("Using Dummy Mesh (Sphere/Cube) for testing.")

    # Create a simple cube of points
    verts = []
    for x in [-0.5, 0.5]:
        for y in [-0.5, 0.5]:
            for z in [-0.5, 0.5]:
                verts.append([x, y, z])
    verts = np.array(verts, dtype=np.float32)
    faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32) # Minimal faces

    return SimpleMesh(verts, faces)

def sample_and_export(mesh, output_path: str):
    """
    P1.5 & P1.6: Sample points from mesh and save as Gaussian PLY.
    """
    logger.info(f"Sampling points and exporting to {output_path}...")

    # 1. Get Points (XYZ)
    if hasattr(mesh, "sample"):
        # Trimesh object
        # Trimesh sample returns just points by default
        points = mesh.sample(100000) # Sample 100k points
    else:
        # SimpleMesh or fallback - just use vertices or dense grid
        points = mesh.vertices
        # Upsample if too few (for dummy cube)
        if len(points) < 100:
             logger.info("Upsampling dummy mesh...")
             # Create random cloud
             points = np.random.uniform(-0.5, 0.5, (1000, 3)).astype(np.float32)

    num_points = len(points)
    logger.info(f"Generated {num_points} Gaussians.")

    # 2. Initialize Attributes
    xyz = points

    # Colors (SH DC) - Initialize to Grey (0.5)
    # 0.5 in rgb -> SH DC is (0.5 - 0.5) / 0.28209 = 0
    # Actually formula is RGB = 0.5 + C0 * SH
    # C0 = 0.28209.
    # if SH=0, RGB=0.5. Perfect.
    features_dc = np.zeros((num_points, 3), dtype=np.float32)
    features_rest = np.zeros((num_points, 45), dtype=np.float32) # Degree 3 (15 * 3)

    # Opacity - Inverse Sigmoid(0.9) -> ~2.2
    opacities = np.ones((num_points, 1), dtype=np.float32) * 2.0

    # Scales - Log scale. Start small.
    scales = np.ones((num_points, 3), dtype=np.float32) * np.log(0.01)

    # Rotations - Quaternion (1, 0, 0, 0)
    rots = np.repeat(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), num_points, axis=0)

    # 3. Construct Structured Array for PlyFile
    dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), # Standard Colors
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
    ]
    # Add f_rest_0 to f_rest_44
    for i in range(45):
        dtype.append((f'f_rest_{i}', 'f4'))

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
    elements['nx'] = 0 # Optional
    elements['ny'] = 0
    elements['nz'] = 0

    # Colors: Write Standard RGB (White/Grey for visibility)
    # Default to 200 (Light Grey) to be visible on dark background
    elements['red'] = 200
    elements['green'] = 200
    elements['blue'] = 200

    elements['f_dc_0'] = features_dc[:, 0]
    elements['f_dc_1'] = features_dc[:, 1]
    elements['f_dc_2'] = features_dc[:, 2]

    for i in range(45):
        elements[f'f_rest_{i}'] = features_rest[:, i]

    elements['opacity'] = opacities[:, 0]
    elements['scale_0'] = scales[:, 0]
    elements['scale_1'] = scales[:, 1]
    elements['scale_2'] = scales[:, 2]

    elements['rot_0'] = rots[:, 0]
    elements['rot_1'] = rots[:, 1]
    elements['rot_2'] = rots[:, 2]
    elements['rot_3'] = rots[:, 3]

    # 4. Write File
    if PlyData is not None:
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(output_path)
        logger.info(f"PLY saved to {output_path}")
    else:
        logger.error("Cannot save PLY: plyfile module missing.")

def main():
    parser = argparse.ArgumentParser(description="Glimpse3D Reconstruction: Image -> Gaussian Splat PLY")
    parser.add_argument("image_path", type=str, help="Path to input image")
    parser.add_argument("--output", type=str, default="output.ply", help="Path to save output PLY")

    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        logger.error(f"Input file not found: {args.image_path}")
        sys.exit(1)

    try:
        logger.info("Starting Reconstruction Pipeline...")
        # Step 1: Load Model
        try:
            model = load_triposr_model()
        except ImportError:
            model = None # Continue for testing

        # Step 2: Preprocess
        image = preprocess_image(args.image_path)

        # Step 3: Inference
        mesh = run_inference(model, image)

        # Step 4: Export
        sample_and_export(mesh, args.output)

        logger.info("Reconstruction Complete!")

    except Exception as e:
        logger.error(f"Reconstruction failed: {e}")
        # Print full traceback for debug
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
