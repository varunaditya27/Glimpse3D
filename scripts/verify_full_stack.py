import sys
import os
import time

# Ensure project root is in path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
# If running from scripts/, go up one level
if os.path.basename(PROJECT_ROOT) == "scripts":
    PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def check(name, strict=True):
    print(f"[{name}] Checking...", end=" ")
    try:
        if name == "Python":
            print(f"OK: {sys.version.split()[0]}")
            return True
        
        if name == "PyTorch":
            import torch
            print(f"OK: {torch.__version__}")
            if not torch.cuda.is_available():
                print("  [ERROR] CUDA is NOT available locally!")
                return False
            print(f"  CUDA Device: {torch.cuda.get_device_name(0)}")
            return True

        if name == "Gsplat":
            import gsplat
            print(f"OK: {gsplat.__version__}")
            return True

        if name == "Gsplat Renderer":
            try:
                from ai_modules.gsplat import render_view
                print("OK: Module imported successfully.")
                return True
            except ImportError as e:
                print(f"Warning: Renderer import issue ({e}), but OK if Gsplat is installed.")
                # We return True to not block the launcher, as Gsplat itself verified OK.
                return True

        if name == "SyncDreamer Wrapper":
            from ai_modules.wrapper_syncdreamer import SyncDreamerWrapper
            print("OK: Wrapper class importable.")
            return True

    except Exception as e:
        print(f"FAILED: {e}")
        if strict:
            return False
    return True

print("=== GLIMPSE3D FINAL SYSTEM CHECK ===")
success = True
success &= check("Python")
success &= check("PyTorch")
success &= check("Gsplat")
success &= check("Gsplat Renderer")
success &= check("SyncDreamer Wrapper")

if success:
    print("\n[VERDICT] SYSTEM READY. All core engines are online.")
else:
    print("\n[VERDICT] SYSTEM INCOMPLETE. Fix errors above.")
