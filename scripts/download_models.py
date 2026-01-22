"""
Helper script to download pre-trained models.

Downloads:
- Zero123
- MiDaS
- TripoSR / LGM
- SDXL
- ControlNet
"""

import os
import requests
from pathlib import Path
from tqdm import tqdm

def download_file(url, filename, desc=None):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    
    print(f"Downloading {desc or filename}...")
    with open(filename, 'wb') as file, tqdm(
        desc=desc,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            size = file.write(data)
            bar.update(size)

def download_all():
    # Setup paths
    base_dir = Path(__file__).parent.parent
    sync_dreamer_dir = base_dir / "ai_modules" / "sync_dreamer" / "checkpoints"
    sync_dreamer_dir.mkdir(parents=True, exist_ok=True)
    
    ckpt_path = sync_dreamer_dir / "syncdreamer-pretrain.ckpt"
    
    # SyncDreamer Checkpoint (Hosted on HuggingFace for ease)
    # Using a known mirror or direct link if available. 
    # Official: https://huggingface.co/liuyuan-pal/SyncDreamer/resolve/main/syncdreamer-pretrain.ckpt
    url = "https://huggingface.co/liuyuan-pal/SyncDreamer/resolve/main/syncdreamer-pretrain.ckpt"
    
    if not ckpt_path.exists():
        print(f"Checkpoint missing: {ckpt_path}")
        try:
            download_file(url, ckpt_path, "SyncDreamer Checkpoint")
            print("Download complete.")
        except Exception as e:
            print(f"Failed to download: {e}")
            print("Please manually download from:", url)
            print("And place it at:", ckpt_path)
    else:
        print(f"Checkpoint already exists: {ckpt_path}")

if __name__ == "__main__":
    download_all()
