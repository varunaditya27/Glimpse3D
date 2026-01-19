import sys
import os
import numpy as np

# Add parent path to allow importing from ai_modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# Try importing utils from ai_modules
try:
    from ai_modules.gsplat.utils_gs import load_ply
except ImportError:
    # Use plyfile directly if utils fail import logic
    from plyfile import PlyData

def compare_ply(chk_name, f1, f2):
    print(f"--- Checking {chk_name} ---")
    try:
        from plyfile import PlyData
        p1 = PlyData.read(f1)['vertex']
        p2 = PlyData.read(f2)['vertex']
        
        diffs = []
        for prop in p1.data.dtype.names:
            d = np.abs(p1[prop] - p2[prop]).mean()
            diffs.append(d)
            if d > 0.0001:
                print(f"Property '{prop}' changed by avg {d:.6f}")
        
        total_diff = sum(diffs)
        if total_diff < 0.00001:
            print("RESULT: Files are IDENTICAL. (No learning happened)")
        else:
            print(f"RESULT: Files are DIFFERENT. (Total Delta: {total_diff:.6f})")
            
    except Exception as e:
        print(f"Comparison failed: {e}")

if __name__ == "__main__":
    base = "ai_modules/gsplat/output/full_run_v2"
    f1 = os.path.join(base, "initial_splat.ply")
    f2 = os.path.join(base, "final_refined.ply")
    
    if os.path.exists(f1) and os.path.exists(f2):
        compare_ply("Run V2", f1, f2)
    else:
        print("Files not found.")
