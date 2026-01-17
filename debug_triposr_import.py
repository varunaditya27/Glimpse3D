import sys
import os
import traceback

# Mimic the path setup in reconstruct.py
current_dir = os.path.dirname(os.path.abspath(__file__))
# current_dir is root (where I will run it)
# ai_modules/gsplat/reconstruct.py does:
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../TripoSR")))
# from root, TripoSR is at ai_modules/TripoSR

triposr_path = os.path.abspath(os.path.join(current_dir, "ai_modules/TripoSR"))
print(f"Adding to path: {triposr_path}")
sys.path.append(triposr_path)

print("Attempting to import tsr...")
try:
    import tsr
    print("Successfully imported tsr!")
    from tsr.system import TSR
    print("Successfully imported TSR class!")
except Exception:
    print("Import Failed!")
    traceback.print_exc()

print("Checking installed packages...")
try:
    import einops
    print("einops: OK")
except ImportError:
    print("einops: MISSING")

try:
    import omegaconf
    print("omegaconf: OK")
except ImportError:
    print("omegaconf: MISSING")
