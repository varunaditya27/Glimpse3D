import torch
import numpy as np
import os
import sys
from plyfile import PlyData, PlyElement

# Add root to sys.path
sys.path.append(os.getcwd())

from ai_modules.gsplat.utils_gs import load_ply, save_ply

GOLD_PATH = "gold_standard.ply"
RESULT_PATH = "reconstructed.ply"

def generate_gold_ply():
    print(f"Generating {GOLD_PATH}...")
    N = 100
    
    # Generate structured array matching gsplat expectation
    # 3 (pos) + 3 (norm) + 3 (dc) + 45 (rest) + 1 (op) + 3 (scale) + 4 (rot) = 62 float32s
    
    dtype_list = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4')
    ]
    
    for i in range(45):
        dtype_list.append((f'f_rest_{i}', 'f4'))
        
    dtype_list.append(('opacity', 'f4'))
    dtype_list.append(('scale_0', 'f4'))
    dtype_list.append(('scale_1', 'f4'))
    dtype_list.append(('scale_2', 'f4'))
    dtype_list.append(('rot_0', 'f4'))
    dtype_list.append(('rot_1', 'f4'))
    dtype_list.append(('rot_2', 'f4'))
    dtype_list.append(('rot_3', 'f4'))
    
    arr = np.zeros(N, dtype=dtype_list)
    
    # Fill with random data
    rng = np.random.default_rng(42) # Fixed seed
    
    arr['x'] = rng.standard_normal(N)
    arr['y'] = rng.standard_normal(N)
    arr['z'] = rng.standard_normal(N)
    
    arr['f_dc_0'] = rng.random(N)
    arr['opacity'] = rng.random(N)
    
    # Random f_rest
    for i in range(45):
        arr[f'f_rest_{i}'] = rng.random(N) * 0.1

    el = PlyElement.describe(arr, 'vertex')
    PlyData([el]).write(GOLD_PATH)
    print("Gold PLY generated.")

def verify_loop():
    if not os.path.exists(GOLD_PATH):
        generate_gold_ply()
        
    print(f"Loading {GOLD_PATH}...")
    model = load_ply(GOLD_PATH)
    
    print(f"Saving to {RESULT_PATH}...")
    save_ply(model, RESULT_PATH)
    
    print("Comparing files...")
    
    # Compare raw data
    ply_gold = PlyData.read(GOLD_PATH)['vertex'].data
    ply_res = PlyData.read(RESULT_PATH)['vertex'].data
    
    # Compare field by field
    success = True
    for name in ply_gold.dtype.names:
        g_data = ply_gold[name]
        r_data = ply_res[name]
        
        # Allow tiny float error due to tensor roundtrips
        if not np.allclose(g_data, r_data, atol=1e-6):
            print(f"FAIL: Mismatch in attribute '{name}'")
            print(f"  Gold: {g_data[:5]}")
            print(f"  Res : {r_data[:5]}")
            success = False
            
    if success:
        print("✅ BINARY/DATA IDENTITY CONFIRMED")
        print("   The IO loop is lossless (within float32 precision).")
    else:
        print("❌ DATA MISMATCH")
        exit(1)

if __name__ == "__main__":
    verify_loop()
