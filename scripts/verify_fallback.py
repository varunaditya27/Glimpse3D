
import os
import sys
import asyncio
import shutil
import traceback
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent / "backend"))

try:
    import trimesh
except ImportError:
    print("Trimesh not installed. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "trimesh"])
    import trimesh

# Define paths
BACKEND_DIR = Path(__file__).parent.parent / "backend"
SERVICE_DIR = BACKEND_DIR / "app" / "services"
GSPLAT_SERVICE = SERVICE_DIR / "gsplat_service.py"
GSPLAT_BAK = SERVICE_DIR / "gsplat_service.py.bak"

async def run_test():
    print("Starting Fallback Verification...")
    
    # 1. Simulate Failure by hiding the service
    if GSPLAT_SERVICE.exists():
        print(f"Renaming {GSPLAT_SERVICE.name} to force fallback...")
        GSPLAT_SERVICE.rename(GSPLAT_BAK)
    else:
        print("Service file not found? Checking backup...")
        if not GSPLAT_BAK.exists():
             print("Warning: Neither service nor backup found.")
    
    try:
        # Import PipelineManager (Lazy import ensures we don't cache the missing module too early if that matters, though here it's local import)
        from app.services.pipeline_manager import PipelineManager
        
        pm = PipelineManager()
        
        # Output file
        output_dir = Path("temp_verification")
        output_dir.mkdir(exist_ok=True)
        img_path = "dummy_image.png" # Not actually read by fallback
        
        # 2. Trigger fallback
        print("Calling _run_coarse_reconstruction...")
        # Note: we expect a warning log here
        model_path, warnings = await pm._run_coarse_reconstruction(img_path, output_dir)
        
        print(f"Result Path: {model_path}")
        print(f"Warnings: {warnings}")
        
        # 3. Verify Output
        if not os.path.exists(model_path):
            raise FileNotFoundError("Fallback did not produce a file!")
            
        print("File created. Verifying with Trimesh...")
        
        try:
            mesh = trimesh.load(model_path)
            print("Trimesh load successful!")
            print(f"Mesh Type: {type(mesh)}")
            
            # Check for vertices (tetrahedron has 4)
            if hasattr(mesh, 'vertices'):
                print(f"Vertices: {len(mesh.vertices)}")
                if len(mesh.vertices) < 4:
                     print("Warning: Mesh has fewer than 4 vertices?")
            
            print("\n✅ SUCCESS: Fallback PLY is valid and loadable.")
            
        except Exception as e:
            print(f"\n❌ FAILURE: Trimesh could not load the file: {e}")
            raise
            
    except Exception as e:
        print(f"\n❌ FAILURE during test execution: {e}")
        traceback.print_exc()
        
    finally:
        # 4. Restore Service
        if GSPLAT_BAK.exists():
            print(f"Restoring {GSPLAT_SERVICE.name}...")
            GSPLAT_BAK.rename(GSPLAT_SERVICE)
        
        # Cleanup
        if Path("temp_verification").exists():
            shutil.rmtree("temp_verification")

if __name__ == "__main__":
    asyncio.run(run_test())
