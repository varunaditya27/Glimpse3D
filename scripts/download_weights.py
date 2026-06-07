"""
Glimpse3D model-weight downloader (idempotent, robust).

Fetches all required weights into the canonical directory that the code
actually loads from: ai_modules/sync_dreamer/ckpt/

Verified against ai_modules/sync_dreamer/inference.py:
    - checkpoint_path = <module>/ckpt/syncdreamer-pretrain.ckpt   (line ~84)
    - clip_path       = <module>/ckpt/ViT-L-14.pt                 (line ~116)

Usage:
    python scripts/download_weights.py                # SyncDreamer ckpt + CLIP
    python scripts/download_weights.py --skip-clip
    python scripts/download_weights.py --prewarm-hf   # also warm HF caches
    python scripts/download_weights.py --dest some/dir
"""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DEST = PROJECT_ROOT / "ai_modules" / "sync_dreamer" / "ckpt"

# Google Drive id for SyncDreamer pretrain checkpoint (~2 GB).
# From inference.py FileNotFoundError hint:
#   https://drive.google.com/file/d/1ypyD5WXxAnsWjnHgAfOAGolV0Zd9kpam/view
SYNCDREAMER_GDRIVE_ID = "1ypyD5WXxAnsWjnHgAfOAGolV0Zd9kpam"
SYNCDREAMER_FILENAME = "syncdreamer-pretrain.ckpt"
SYNCDREAMER_MIN_BYTES = 1_000_000_000  # > 1 GB; smaller => likely an HTML interstitial

# CLIP image encoder used by SyncDreamer (config.model.params.clip_image_encoder_path).
# Mirrored on HuggingFace alongside the SyncDreamer assets.
CLIP_HF_REPO = "camenduru/SyncDreamer"
CLIP_FILENAME = "ViT-L-14.pt"

# HuggingFace repos used elsewhere in the pipeline. These auto-download on first
# use, so prewarming is OPTIONAL -- it just avoids a stall mid-pipeline on the
# first real Colab run.
#   - TripoSR mesh model:        ai_modules/TripoSR
#   - SDXL-Lightning UNet + base SDXL: ai_modules/diffusion/sdxl_lightning.py
#       repo "ByteDance/SDXL-Lightning" (UNet) + "stabilityai/stable-diffusion-xl-base-1.0"
#   - ControlNet depth (default "xinsir_depth"): ai_modules/diffusion/controlnet_depth.py
#       repo "xinsir/controlnet-depth-sdxl-1.0"
PREWARM_HF_REPOS = [
    "stabilityai/TripoSR",
    "ByteDance/SDXL-Lightning",
    "stabilityai/stable-diffusion-xl-base-1.0",
    "xinsir/controlnet-depth-sdxl-1.0",
]


def _human_size(num_bytes):
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"


def download_syncdreamer(dest_dir):
    """Download the SyncDreamer pretrain checkpoint via gdown.

    gdown is required (not plain requests) because Google Drive serves a
    confirm-token interstitial for large files; requests would silently save
    the HTML page instead of the checkpoint.
    """
    target = dest_dir / SYNCDREAMER_FILENAME

    if target.exists():
        size = target.stat().st_size
        if size >= SYNCDREAMER_MIN_BYTES:
            print(f"[SyncDreamer] OK, already present: {target} ({_human_size(size)})")
            return True
        print(
            f"[SyncDreamer] Found too-small file ({_human_size(size)}); "
            "re-downloading."
        )
        target.unlink()

    try:
        import gdown
    except ImportError:
        print(
            "[SyncDreamer] ERROR: gdown not installed. "
            "Run: pip install gdown",
            file=sys.stderr,
        )
        return False

    url = f"https://drive.google.com/uc?id={SYNCDREAMER_GDRIVE_ID}"
    print(f"[SyncDreamer] Downloading checkpoint (~2 GB) from {url}")
    print(f"[SyncDreamer]   -> {target}")
    gdown.download(url, str(target), quiet=False)

    if not target.exists():
        print("[SyncDreamer] ERROR: download produced no file.", file=sys.stderr)
        return False

    size = target.stat().st_size
    if size < SYNCDREAMER_MIN_BYTES:
        # Almost certainly an HTML interstitial / quota page -- delete and fail loud.
        target.unlink()
        raise RuntimeError(
            f"[SyncDreamer] Downloaded file is only {_human_size(size)} "
            f"(< 1 GB). It is likely an HTML interstitial, not the checkpoint. "
            f"Deleted it. Try again later (Drive quota) or download manually "
            f"from https://drive.google.com/file/d/{SYNCDREAMER_GDRIVE_ID}/view"
        )

    print(f"[SyncDreamer] Done: {target} ({_human_size(size)})")
    return True


def download_clip(dest_dir):
    """Download the CLIP ViT-L-14.pt encoder via huggingface_hub.

    This is the previously-missing file: inference.py wires it into the
    SyncDreamer config when ckpt/ViT-L-14.pt exists.
    """
    target = dest_dir / CLIP_FILENAME

    if target.exists():
        print(
            f"[CLIP] OK, already present: {target} "
            f"({_human_size(target.stat().st_size)})"
        )
        return True

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print(
            "[CLIP] ERROR: huggingface_hub not installed. "
            "Run: pip install huggingface_hub",
            file=sys.stderr,
        )
        return False

    print(f"[CLIP] Downloading {CLIP_FILENAME} from HF repo '{CLIP_HF_REPO}'")
    print(f"[CLIP]   -> {target}")
    # local_dir places the file directly at dest/ViT-L-14.pt (no nested cache dirs).
    hf_hub_download(
        repo_id=CLIP_HF_REPO,
        filename=CLIP_FILENAME,
        local_dir=str(dest_dir),
    )

    if not target.exists():
        print(
            f"[CLIP] ERROR: expected {target} after download but it is missing.",
            file=sys.stderr,
        )
        return False

    print(f"[CLIP] Done: {target} ({_human_size(target.stat().st_size)})")
    return True


def prewarm_hf():
    """Warm HuggingFace caches for the downstream pipeline repos.

    Uses snapshot_download so the full repo lands in the HF cache and first
    real use does not stall. These models otherwise auto-download on first use,
    so failures here are non-fatal (logged, not raised).
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "[prewarm] ERROR: huggingface_hub not installed. "
            "Run: pip install huggingface_hub",
            file=sys.stderr,
        )
        return False

    all_ok = True
    for repo_id in PREWARM_HF_REPOS:
        print(f"[prewarm] snapshot_download('{repo_id}') ...")
        try:
            path = snapshot_download(repo_id=repo_id)
            print(f"[prewarm]   cached at {path}")
        except Exception as exc:  # non-fatal: these auto-download on first use
            all_ok = False
            print(f"[prewarm]   WARNING: failed to prewarm '{repo_id}': {exc}")
    return all_ok


def summarize(dest_dir):
    print("\n=== Summary ===")
    print(f"Destination: {dest_dir}")
    for name in (SYNCDREAMER_FILENAME, CLIP_FILENAME):
        path = dest_dir / name
        if path.exists():
            print(f"  [present] {name} ({_human_size(path.stat().st_size)})")
        else:
            print(f"  [MISSING] {name}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Download Glimpse3D model weights into the canonical ckpt dir."
    )
    parser.add_argument(
        "--skip-syncdreamer",
        action="store_true",
        help="Skip the SyncDreamer pretrain checkpoint download.",
    )
    parser.add_argument(
        "--skip-clip",
        action="store_true",
        help="Skip the CLIP ViT-L-14.pt download.",
    )
    parser.add_argument(
        "--prewarm-hf",
        action="store_true",
        help="Also prewarm HF caches for TripoSR / SDXL-Lightning / ControlNet.",
    )
    parser.add_argument(
        "--dest",
        default=str(DEFAULT_DEST),
        help=f"Destination directory (default: {DEFAULT_DEST}).",
    )
    args = parser.parse_args(argv)

    dest_dir = Path(args.dest).resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    print("Glimpse3D weight downloader")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Destination : {dest_dir}\n")

    ok = True

    if args.skip_syncdreamer:
        print("[SyncDreamer] Skipped (--skip-syncdreamer).")
    else:
        ok = download_syncdreamer(dest_dir) and ok

    if args.skip_clip:
        print("[CLIP] Skipped (--skip-clip).")
    else:
        ok = download_clip(dest_dir) and ok

    if args.prewarm_hf:
        # Prewarm failures are non-fatal and must not flip the exit code.
        prewarm_hf()

    summarize(dest_dir)

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
