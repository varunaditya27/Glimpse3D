"""
Thin shim: delegates to scripts/download_weights.py.

Historically this script downloaded into the WRONG directory
(ai_modules/sync_dreamer/checkpoints/) and used a 401-gated HF URL.
The canonical, idempotent downloader is now download_weights.py, which
fetches into ai_modules/sync_dreamer/ckpt/ (the dir inference.py loads from).

Kept as a stable entry point to avoid breaking anything that calls it.
"""

import sys
from pathlib import Path

# Ensure this script's directory is importable when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from download_weights import main

if __name__ == "__main__":
    sys.exit(main())
