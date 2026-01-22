import sys
import os
import torch
import gc

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ai_modules.sync_dreamer.inference import generate_multiview, cleanup as cleanup_sync
from ai_modules.diffusion import EnhanceService, EnhanceConfig

def test_syncdreamer():
    print("Testing SyncDreamer...")
    try:
        # Create a dummy image
        from PIL import Image
        img = Image.new('RGB', (512, 512), color='red')
        img.save("debug_input.png")
        
        # Run generation
        print("  Generating views...", end=" ", flush=True)
        generate_multiview("debug_input.png", "output_debug_sync", batch_view_num=1)
        print("OK")
        
        print("  Cleaning up...", end=" ", flush=True)
        cleanup_sync()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("OK")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False

def test_diffusion():
    print("\nTesting Diffusion (SDXL)...")
    try:
        print("  Loading EnhanceService...", end=" ", flush=True)
        config = EnhanceConfig.for_t4_gpu()
        service = EnhanceService(config=config)
        service.load()
        print("OK")
        
        print("  Running enhancement...", end=" ", flush=True)
        service.enhance("debug_input.png", "test prompt")
        print("OK")
        
        print("  Cleaning up...", end=" ", flush=True)
        service.unload()
        del service
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("OK")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False

if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("CUDA NOT AVAILABLE")

    if not test_syncdreamer():
        print("SyncDreamer caused the crash.")
        sys.exit(1)
        
    if not test_diffusion():
        print("SDXL caused the crash.")
        sys.exit(1)
        
    print("\nALL MODULES PASSED ISOLATION TEST.")
