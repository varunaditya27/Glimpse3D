
import os
import requests
import sys
from pathlib import Path

def download_file(url, func_dest):
    if os.path.exists(func_dest):
        print(f"File exists: {func_dest} (Skipping)")
        return

    print(f"Downloading {url} to {func_dest}...")
    try:
        os.makedirs(os.path.dirname(func_dest), exist_ok=True)
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(func_dest, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Done.")
    except Exception as e:
        print(f"Failed to download: {e}")

PROJECT_ROOT = Path(__file__).parent.parent
print(f"Project Root: {PROJECT_ROOT}")

# 1. SyncDreamer Weights (~2GB)
# Official link or huggingface link
# SyncDreamer uses 'syncdreamer-pretrain.ckpt'
# HF Link failed (401), using Google Drive mirror
syncdreamer_path = PROJECT_ROOT / "ai_modules" / "sync_dreamer" / "ckpt" / "syncdreamer-pretrain.ckpt"

print("\n--- Downloading Model Weights ---")
print(f"Target: {syncdreamer_path}")
print("Note: This file is ~2GB. Downloading automatically from Google Drive...")

try:
    import gdown
    # Google Drive ID from the error message url
    file_id = "1ypyD5WXxAnsWjnHgAfOAGolV0Zd9kpam"
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, str(syncdreamer_path), quiet=False)
    print("Download complete.")
except ImportError:
    print("Error: gdown not installed. Please run 'pip install gdown'")
except Exception as e:
    print(f"Download failed: {e}")

print("\nSetup Complete.")
