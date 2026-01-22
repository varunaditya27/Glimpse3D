import requests
import time
import os

BASE_URL = "http://localhost:8000"

def test_full_flow():
    print(f"Testing connectivity to {BASE_URL}...")
    try:
        r = requests.get(f"{BASE_URL}/docs")
        if r.status_code == 200:
            print("Backend is online/reachable.")
        else:
            print(f"Backend returned {r.status_code}")
    except Exception as e:
        print(f"Backend not reachable: {e}")
        return

    # 1. Upload
    print("\n[1] Uploading Image...")
    # Using the input image we verified earlier
    image_path = r"e:\SEM 3\EL\Glimpse3D\assets\input_images\input.png"
    if not os.path.exists(image_path):
        print(f"Error: {image_path} does not exist.")
        return

    files = {'file': open(image_path, 'rb')}
    try:
        r_upload = requests.post(f"{BASE_URL}/upload/", files=files)
        print("Upload Response:", r_upload.json())
        
        if r_upload.status_code != 200 or not r_upload.json().get('success'):
            print("Upload failed.")
            return
            
        uploaded_path = r_upload.json()['file_path']
        print(f"Image uploaded to: {uploaded_path}")
        
    except Exception as e:
        print(f"Upload failed: {e}")
        return

    # 2. Generate
    print("\n[2] Triggering Generation Pipeline...")
    payload = {
        "image_path": uploaded_path,
        # output_dir optional, backend will create temp
    }
    
    try:
        r_gen = requests.post(f"{BASE_URL}/generate/", json=payload)
        print("Generate Response:", r_gen.json())
        
        if r_gen.status_code != 200:
            print("Generation trigger failed.")
            return
            
        job_id = r_gen.json()['job_id']
        print(f"Job ID: {job_id}")
        
    except Exception as e:
        print(f"Generate trigger failed: {e}")
        return

    # 3. Poll Status
    print("\n[3] Polling Job Status...")
    status = "starting"
    while status not in ["completed", "failed"]:
        try:
            r_status = requests.get(f"{BASE_URL}/generate/status/{job_id}")
            data = r_status.json()
            status = data['status']
            progress = data.get('progress', 0)
            print(f"Status: {status} (Progress: {progress*100:.1f}%)")
            
            if status == "failed":
                print("Generation FAILED.")
                print("Error:", data.get('error'))
                return
                
            if status == "completed":
                print("Generation COMPLETED!")
                print("Result:", data.get('result'))
                break
                
            time.sleep(2)
        except Exception as e:
            print(f"Polling failed: {e}")
            break

if __name__ == "__main__":
    test_full_flow()
