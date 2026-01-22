#!/usr/bin/env python3
"""
Test script to verify backend-frontend connection and basic functionality.
"""

import requests
import json
import time
import os
from pathlib import Path

# Backend URL
BACKEND_URL = "http://localhost:8000"

def test_backend_health():
    """Test if backend is running."""
    try:
        response = requests.get(f"{BACKEND_URL}/")
        if response.status_code == 200:
            print("✅ Backend is running")
            return True
        else:
            print(f"❌ Backend returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to backend: {e}")
        return False

def test_upload():
    """Test file upload."""
    # Use a sample image
    sample_images = [
        "assets/sample_inputs/image.png",
        "assets/input_images/input.png"
    ]

    image_path = None
    for img in sample_images:
        if os.path.exists(img):
            image_path = img
            break

    if not image_path:
        print("❌ No sample image found")
        return None

    print(f"📤 Testing upload with {image_path}")

    try:
        with open(image_path, 'rb') as f:
            files = {'file': ('test_image.png', f, 'image/png')}
            response = requests.post(f"{BACKEND_URL}/upload/", files=files)

        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print("✅ Upload successful")
                print(f"   File path: {data['file_path']}")
                return data['file_path']
            else:
                print(f"❌ Upload failed: {data.get('error')}")
                return None
        else:
            print(f"❌ Upload HTTP error: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Upload exception: {e}")
        return None

def test_generation(image_path):
    """Test 3D generation."""
    print(f"🚀 Testing generation with {image_path}")

    try:
        payload = {"image_path": image_path}
        response = requests.post(
            f"{BACKEND_URL}/generate/",
            json=payload,
            headers={'Content-Type': 'application/json'}
        )

        if response.status_code == 200:
            data = response.json()
            if data.get('success') and data.get('job_id'):
                job_id = data['job_id']
                print(f"✅ Generation started, job ID: {job_id}")
                return job_id
            else:
                print(f"❌ Generation failed to start: {data}")
                return None
        else:
            print(f"❌ Generation HTTP error: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Generation exception: {e}")
        return None

def test_status_polling(job_id):
    """Test status polling."""
    print(f"📊 Testing status polling for job {job_id}")

    max_attempts = 30  # 30 seconds timeout
    for attempt in range(max_attempts):
        try:
            response = requests.get(f"{BACKEND_URL}/generate/status/{job_id}")

            if response.status_code == 200:
                data = response.json()
                status = data.get('status')
                progress = data.get('progress', 0)

                print(f"   Status: {status}, Progress: {progress:.1f}")
                if status == 'completed':
                    print("✅ Generation completed!")
                    if data.get('result') and data['result'].get('model_url'):
                        model_url = data['result']['model_url']
                        print(f"   Model URL: {model_url}")

                        # Test if model file is accessible
                        full_url = f"{BACKEND_URL}{model_url}"
                        model_response = requests.head(full_url)
                        if model_response.status_code == 200:
                            print("✅ Model file is accessible")
                        else:
                            print(f"❌ Model file not accessible: {model_response.status_code}")

                    return True

                elif status == 'failed':
                    print(f"❌ Generation failed: {data.get('error')}")
                    return False

            else:
                print(f"❌ Status check HTTP error: {response.status_code}")
                return False

        except Exception as e:
            print(f"❌ Status check exception: {e}")
            return False

        time.sleep(1)

    print("⏰ Status polling timeout")
    return False

def main():
    """Run all tests."""
    print("🧪 Glimpse3D Backend Connection Test")
    print("=" * 40)

    # Test 1: Backend health
    if not test_backend_health():
        return

    # Test 2: Upload
    image_path = test_upload()
    if not image_path:
        return

    # Test 3: Generation
    job_id = test_generation(image_path)
    if not job_id:
        return

    # Test 4: Status polling
    success = test_status_polling(job_id)

    print("\n" + "=" * 40)
    if success:
        print("🎉 All backend tests passed!")
        print("\nNext steps:")
        print("1. Restart frontend: cd frontend && npm run dev")
        print("2. Upload image in frontend - should now work!")
    else:
        print("❌ Some tests failed. Check backend logs for details.")

if __name__ == "__main__":
    main()
