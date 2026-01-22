import os
from pathlib import Path

def check_manual_output():
    file_path = Path("e:/SEM 3/EL/Glimpse3D/assets/outputs/manual_test_result.ply")
    if file_path.exists():
        size = file_path.stat().st_size
        print(f"File: {file_path}")
        print(f"Size: {size} bytes")
        
        if size < 5000:
            print("WARNING: File is dangerously small. Likely empty/dummy.")
            # Read first few lines
            with open(file_path, 'r', errors='ignore') as f:
                print("--- Head ---")
                print(f.read(300))
                print("--- End Head ---")
    else:
        print(f"File not found: {file_path}")

if __name__ == "__main__":
    check_manual_output()
