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

- `POST /upload`: Upload source image.
- `POST /generate/{id}`: Trigger initial coarse 3D reconstruction.
- `POST /refine/{id}`: Run the **Enhancement Loop** (SDXL + Backprojection).
- `GET /export/{id}`: Download the final 3D model.
