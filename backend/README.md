# ⚙️ Glimpse3D Backend

The central orchestration engine for Glimpse3D, built with **FastAPI** and **PyTorch**. It manages the entire 3D generation pipeline, from image upload to final model export.

## 🧠 Responsibilities

- **Pipeline Orchestration**: Coordinates data flow between Zero123, MiDaS, SDXL, and Gaussian Splatting modules.
- **State Management**: Tracks the status of uploads, generation jobs, and refinement iterations.
- **API Layer**: Provides REST endpoints for the frontend to interact with the system.
- **GPU Resource Management**: Efficiently loads and unloads heavy AI models.

## 📂 Directory Structure

```
backend/
├── app/
│   ├── routes/         # API Endpoints (Upload, Generate, Refine, Export)
│   ├── services/       # Business Logic & Model Wrappers
│   ├── models/         # Pydantic Schemas (Request/Response)
│   ├── core/           # Config, Logging, Utils
│   └── main.py         # Application Entrypoint
├── tests/              # Integration Tests
├── Dockerfile          # Container Definition
└── requirements.txt    # Python Dependencies
```

## 🚀 Getting Started

### 1. Install Dependencies

Ensure you have Python 3.10+ and CUDA installed.

```bash
pip install -r requirements.txt
```

### 2. Run the Server

Start the FastAPI development server:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.
Interactive docs are at `http://localhost:8000/docs`.

## 🔌 Key Endpoints

- `POST /upload`: Upload and validate source image (size/format/content checks).
- `POST /generate/`: Trigger initial coarse 3D reconstruction (runs in the background; accepts an optional `client_id` for targeted progress updates).
- `GET /generate/status/{job_id}`: Poll the status, progress, result, and warnings of a generation job.
- `POST /refine/`: Run the **Enhancement Loop** (render → SDXL enhance → back-projection / optimize splats).
- `GET /export/{model_id}`: Download the final 3D model (`?format=ply|glb|obj`).
- `GET /compare/projects`: List completed projects available for A/B comparison.
- `GET /compare/{id1}/{id2}`: Fetch two completed projects for side-by-side comparison.
- `WS /ws/{client_id}`: WebSocket channel for real-time, per-client job progress and completion events.

> A `POST /generate/{upload_id}` legacy route is retained for backward compatibility.

## ⚙️ Configuration

The backend reads configuration from environment variables (see [`.env.example`](.env.example) — copy it to `.env` and adjust). All values are optional; defaults are shown.

| Variable | Default | Purpose |
|----------|---------|---------|
| `PERSISTENCE_BACKEND` | `sqlite` | Job repository backend: `sqlite` or `supabase`. |
| `DATABASE_URL` | `sqlite:///./glimpse3d.db` | SQLAlchemy database URL (used when `PERSISTENCE_BACKEND=sqlite`). |
| `SUPABASE_URL` | _(empty)_ | Supabase project REST URL (only when `PERSISTENCE_BACKEND=supabase`). |
| `SUPABASE_SERVICE_ROLE_KEY` | _(empty)_ | Server-side service-role key for all reads/writes. Keep secret; never ship to the browser. |
| `SUPABASE_ANON_KEY` | _(empty)_ | Client-read-only anon key (safe to expose to the frontend, subject to RLS). Not used by the server for writes. |
| `GLIMPSE3D_DEMO_MODE` | _(empty)_ | Skip heavy model inference and return canned outputs (`1`/`true`/`yes` to enable) — useful for a no-GPU smoke test. |
| `PIPELINE_TIMEOUT` | `2700` | Overall pipeline wall-clock budget, in seconds. |
| `GSPLAT_TRAIN_TIMEOUT` | `1800` | Gaussian-splat training step timeout, in seconds. |
| `RENDER_TIMEOUT` | `60` | Per-render timeout, in seconds. |
| `MVCRM_ENABLED` | `true` | Toggle the MVCRM multi-view consistent refinement stage (`1`/`true`/`yes` to enable). |
| `PORT` | `8000` | Port for uvicorn when running `app.main` directly. |
