import requests
import os
from pathlib import Path

BASE_URL = "http://localhost:8001"

import time

def check_outputs():
    # 1. Trigger Generation
    print("Triggering generation...")
    input_image = Path("e:/SEM 3/EL/Glimpse3D/assets/input_images/input.png")
    if not input_image.exists():
        print("Input image not found!")
        return

    url = f"{BASE_URL}/generate/"
    url = f"{BASE_URL}/generate/"
    try:
        # P1.5 Update: Endpoint expects JSON, not files
        payload = {"image_path": str(input_image)}
        # Fix path for Windows (escape backslashes if needed, but requests handles json)
        
        r = requests.post(url, json=payload)
        print(f"Trigger Status: {r.status_code}")
        if r.status_code != 200:
            print(f"Error: {r.text}")
            return
        job_data = r.json()
        job_id = job_data.get("job_id")
        print(f"Job ID: {job_id}")
    except Exception as e:
        print(f"Trigger failed: {e}")
        return

    # 2. Poll Status
    print("Polling for completion...")
    status_url = f"{BASE_URL}/generate/status/{job_id}"
    for _ in range(60): # Wait up to 60s
        try:
            r = requests.get(status_url)
            status_data = r.json()
            status = status_data.get("status")
            print(f"Status: {status}")
            if status == "completed":
                break
            if status == "failed":
                print("Job Failed!")
                return
            time.sleep(1)
        except Exception as e:
            print(f"Polling failed: {e}")
            time.sleep(1)
    
    # 3. List output using job_id
    project_root = Path("e:/SEM 3/EL/Glimpse3D")
    job_dir = project_root / "assets" / "outputs" / job_id
    
    print(f"\nInspecting job dir: {job_dir}")
    if not job_dir.exists():
        print("Job directory not created!")
        return

    files = list(job_dir.iterdir())
    print(f"Files in job dir: {[f.name for f in files]}")
    
    ply_files = [f for f in files if f.suffix == '.ply']
    if not ply_files:
        print("ERROR: No .ply file found in job directory.")
    else:
        target_ply = ply_files[0]
        print(f"Found PLY: {target_ply.name}")
        size = target_ply.stat().st_size
        print(f"Size: {size} bytes")
        
        if size > 100000:
            print("SUCCESS: PLY file size indicates valid mesh!")
        else:
            print("WARNING: PLY file size is suspiciously small.")

        # 4. Try to download via API
        url = f"{BASE_URL}/outputs/{job_id}/{target_ply.name}"
        print(f"\nTesting URL: {url}")
        try:
            r = requests.get(url)
            print(f"Status Code: {r.status_code}")
            if r.status_code == 200:
                print("Server accessible: YES")
            else:
                print("Server accessible: NO")
        except Exception as e:
            print(f"Request failed: {e}")

if __name__ == "__main__":
    check_outputs()
