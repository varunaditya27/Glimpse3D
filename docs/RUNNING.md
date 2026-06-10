# 🏃 Running Glimpse3D

This guide covers two ways to run the app:

- **(A) Local-only smoke** — backend in mock/CPU mode (optionally with `GLIMPSE3D_DEMO_MODE` for a no-GPU UI walkthrough) plus the frontend dev server. No GPU, no weights.
- **(B) Full GPU run** — the GPU-heavy backend on a free Colab GPU exposed via a public tunnel, with the frontend running locally against it. This is where the generative models produce real output.

The real deployment is **split**: the backend runs where the GPU is (Colab), the frontend runs on your machine. Both flows below reflect the actual code — endpoint paths, env vars, and notebook steps are cross-checked against the repo.

---

## Prerequisites

- **Python 3.10+** for the backend.
- **Node.js + npm** for the frontend.
- For full GPU output: a **Google Colab GPU runtime** (T4 is enough). Local CPU is fine for smoke / UI verification, but the heavy models (TripoSR → SyncDreamer → SDXL → gsplat) need a GPU to produce real results.

---

## (A) Local-only smoke (no GPU)

Use this to walk through the UI, exercise the API, and verify the wiring without a GPU or downloaded weights.

### 1. Backend (CPU / demo)

Install the backend dependencies and start the server. From the repo root:

```bash
pip install -r backend/requirements.txt

# Option 1 — run as a module from the repo root (binds 0.0.0.0 and reads PORT, default 8000):
python -m backend.app.main

# Option 2 — from inside backend/ with autoreload:
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API is then at `http://localhost:8000` (interactive docs at `http://localhost:8000/docs`, health check at `GET /`).

**No-GPU UI walkthrough:** set `GLIMPSE3D_DEMO_MODE` to skip heavy model inference and return canned outputs, so the full upload → generate → view flow runs end to end without a GPU:

```bash
# Windows (PowerShell)
$env:GLIMPSE3D_DEMO_MODE = "1"; python -m backend.app.main

# macOS / Linux
GLIMPSE3D_DEMO_MODE=1 python -m backend.app.main
```

Persistence defaults to **SQLite** (`DATABASE_URL=sqlite:///./glimpse3d.db`), so jobs are tracked and survive restarts with no extra setup.

### 2. Frontend (dev server)

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Leave `frontend/.env` empty (or omit `VITE_API_URL`) for local development — the app falls back to `http://localhost:8000` and derives the WebSocket URL automatically. Open the URL Vite prints (usually `http://localhost:5173`).

### 3. Quick API check

You can hit the API directly to confirm it's live:

```bash
curl http://localhost:8000/                      # {"message": "Glimpse3D Backend is running"}
curl http://localhost:8000/compare/projects      # {"projects": [...]}
```

Key routes: `POST /upload`, `POST /generate/`, `GET /generate/status/{job_id}`, `POST /refine/`, `GET /export/{model_id}`, `GET /compare/projects`, `GET /compare/{id1}/{id2}`, and the WebSocket `/ws/{client_id}`.

---

## (B) Full GPU run (Colab backend + local frontend)

This is the real deployment. The backend runs on a Colab GPU and is reached over a public Cloudflare tunnel; the frontend runs locally and points at that tunnel.

### 1. Start the backend on Colab

Open [`notebooks/Glimpse3D_Serve_Backend.ipynb`](../notebooks/Glimpse3D_Serve_Backend.ipynb) in Colab, select a **GPU runtime** (Runtime → Change runtime type → GPU), and **Run all**. The notebook cells, in order:

1. **`nvidia-smi`** — confirm a GPU is attached.
2. **Clone & `cd`** — clone the repo into `/content/Glimpse3D` (skipped if already cloned) and `cd` in.
3. **Install deps** — `requirements-colab.txt` + `backend/requirements.txt`, plus the Cloudflare tunnel helper (`pycloudflared`). Colab already ships a CUDA-enabled `torch`, so it is not reinstalled.
4. **Download weights** — `python scripts/download_weights.py --prewarm-hf` (~6 GB total; idempotent, safe to re-run). See [Model weights](#model-weights) below.
5. **Launch backend** — starts `python -m backend.app.main` in the background on `0.0.0.0:8000`.
6. **Open tunnel** — starts a Cloudflare quick tunnel and **prints the public HTTPS URL**. (Quick tunnels forward WebSocket / `wss` automatically, so realtime `/ws` works through the same URL.)

Keep the Colab tab running — the tunnel and backend stay alive only while the session is active.

### 2. Start the frontend locally

On your own machine, point the frontend at the tunnel URL from the notebook and run the dev server:

```bash
cd frontend
npm install
```

Create / edit `frontend/.env`:

```
VITE_API_URL=https://your-tunnel.trycloudflare.com
```

(Use the exact URL the notebook printed.) Then:

```bash
npm run dev
```

Open the local URL Vite prints (usually `http://localhost:5173`). The app now sends all API and WebSocket traffic to the GPU-backed Colab backend through the tunnel — it derives the WebSocket URL automatically (`https` → `wss`) and resolves backend-relative asset paths (e.g. `/outputs/...`) against `VITE_API_URL`.

### Optional: full run on your own GPU machine

If you have a local CUDA GPU, you can run the backend locally instead of Colab. Download the weights first (next section), then launch the backend exactly as in flow (A) but **without** `GLIMPSE3D_DEMO_MODE`. Leave `frontend/.env` empty so the frontend talks to `http://localhost:8000`.

---

## Model weights

The Colab serve notebook runs this for you (step 4). To fetch weights manually:

```bash
python scripts/download_weights.py             # SyncDreamer checkpoint + CLIP ViT-L-14
python scripts/download_weights.py --prewarm-hf  # also warm the HF caches (TripoSR / SDXL-Lightning / ControlNet-depth)
```

Useful flags: `--skip-syncdreamer`, `--skip-clip`, `--dest <dir>`. Weights land in the canonical directory the code loads from (`ai_modules/sync_dreamer/ckpt/`). The downloader is idempotent — it skips files that are already present and valid. The HF prewarm step is optional (those models otherwise auto-download on first use); prewarm failures are non-fatal.

---

## Environment variables

Backend configuration comes from environment variables. Copy [`backend/.env.example`](../backend/.env.example) to `backend/.env` and adjust. All values are optional; defaults are shown.

| Variable | Default | Purpose |
|----------|---------|---------|
| `PERSISTENCE_BACKEND` | `sqlite` | Job repository backend: `sqlite` or `supabase`. |
| `DATABASE_URL` | `sqlite:///./glimpse3d.db` | SQLAlchemy database URL (used when `PERSISTENCE_BACKEND=sqlite`). |
| `SUPABASE_URL` | _(empty)_ | Supabase project REST URL (only when `PERSISTENCE_BACKEND=supabase`). |
| `SUPABASE_SERVICE_ROLE_KEY` | _(empty)_ | Server-side service-role key. Keep secret; never ship to the browser. |
| `SUPABASE_ANON_KEY` | _(empty)_ | Client-read-only anon key (subject to RLS). Not used by the server for writes. |
| `GLIMPSE3D_DEMO_MODE` | _(empty)_ | Skip heavy inference and return canned outputs (`1`/`true`/`yes`) — for a no-GPU smoke test. |
| `PIPELINE_TIMEOUT` | `2700` | Overall pipeline wall-clock budget, in seconds. |
| `GSPLAT_TRAIN_TIMEOUT` | `1800` | Gaussian-splat training step timeout, in seconds. |
| `RENDER_TIMEOUT` | `60` | Per-render timeout, in seconds. |
| `MVCRM_ENABLED` | `true` | Toggle the MVCRM multi-view consistent refinement stage. |
| `PORT` | `8000` | Port for uvicorn when running `app.main` directly. |

Frontend configuration is a single variable in `frontend/.env`:

| Variable | Default | Purpose |
|----------|---------|---------|
| `VITE_API_URL` | _(empty → `http://localhost:8000`)_ | Base URL of the backend API. Set to the Colab tunnel URL for the full GPU run; leave empty for local dev. |

---

## Status note

The code is complete and **CPU-verified** (frontend build, backend API tests, MVCRM CPU round-trip). Generative visual quality is validated on a **Colab T4 GPU run** — flow (B) above is where the heavy models produce their real output.
