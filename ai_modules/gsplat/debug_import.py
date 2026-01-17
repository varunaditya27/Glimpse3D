import sys
import os
import traceback

# Add TripoSR to path
triposr_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../TripoSR"))
sys.path.append(triposr_path)
print(f"Added path: {triposr_path}")

print("Attempting to import tsr...")
try:
    import tsr
    print(f"tsr imported: {tsr}")
    from tsr.system import TSR
    print(f"TSR imported: {TSR}")
except Exception:
    traceback.print_exc()
