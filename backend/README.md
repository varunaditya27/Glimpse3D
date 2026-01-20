# ⚙️ Glimpse3D Backend

The central orchestration engine for Glimpse3D, built with **FastAPI** and **PyTorch**. It manages the entire 3D generation pipeline from image upload to final model export, with full **Supabase integration** for database tracking and storage.

---

## 🧠 Overview

**Glimpse3D Backend** coordinates the complete pipeline:

1. **Image Upload & Preprocessing** → Background removal
2. **SyncDreamer** → Multi-view generation (16 views)
3. **MiDaS** → Depth estimation for each view
4. **Gaussian Splatting** → Initial 3D reconstruction
5. **SDXL + MVCRM** → Iterative refinement loop
6. **Export** → Convert to .ply, .splat, .glb, .obj formats

All steps are tracked in **Supabase** (database + storage), enabling real-time progress monitoring and full reproducibility.

---

## 📂 Directory Structure

```
backend/
├── app/
│   ├── routes/              # API Endpoints
│   │   ├── upload.py        # Image upload & preprocessing
│   │   ├── generate.py      # Multi-view → Depth → 3D reconstruction
│   │   ├── refine.py        # SDXL enhancement → MVCRM refinement
│   │   ├── export.py        # Format conversion & export
│   │   └── status.py        # Real-time status polling
│   ├── services/            # Business Logic & Model Wrappers
│   │   ├── pipeline_manager.py
│   │   ├── diffusion_service.py
│   │   ├── depth_service.py
│   │   └── gsplat_service.py
│   ├── models/              # Pydantic Schemas (Request/Response)
│   │   ├── request_models.py
│   │   └── response_models.py
│   ├── core/                # Core Utilities
│   │   ├── config.py        # Configuration settings
│   │   ├── logger.py        # Logging utilities
│   │   ├── supabase_client.py  # Supabase client singleton
│   │   ├── database.py      # Database helper functions
│   │   └── storage.py       # Storage upload helpers
│   └── main.py              # FastAPI Application Entrypoint
├── tests/                   # Integration Tests
│   └── test_api.py
├── supabase_schema.sql      # Database schema (run this in Supabase)
├── test_integration.py      # Test script for Supabase setup
├── .env.example             # Environment variable template
├── .gitignore
├── Dockerfile
├── requirements.txt
├── README.md                # This file
├── SUPABASE_SETUP.md        # Detailed setup guide
├── BUCKET_POLICIES.md       # Storage bucket policies
├── QUICKSTART.md            # Quick start guide
├── PIPELINE_DIAGRAM.md      # Visual pipeline diagram
└── IMPLEMENTATION_SUMMARY.md # Implementation details
```

---

## 🚀 Quick Start

### **Prerequisites**

- Python 3.10+
- CUDA (for GPU acceleration)
- Supabase account (free tier works)

### **1. Install Dependencies**

```bash
cd backend
pip install -r requirements.txt
```

### **2. Set Up Supabase**

Follow the detailed guide in **[SUPABASE_SETUP.md](SUPABASE_SETUP.md)**:

1. Create Supabase project
2. Copy `.env.example` → `.env` and add credentials
3. Run `supabase_schema.sql` in Supabase SQL Editor
4. Create 6 storage buckets (see guide)
5. Apply bucket policies

**Quick .env setup:**
```bash
cp .env.example .env
# Edit .env and add:
# SUPABASE_URL=https://xxxxx.supabase.co
# SUPABASE_ANON_KEY=eyJhbGciOiJIUzI...
```

### **3. Test Integration**

```bash
python test_integration.py
```

This will verify:
- ✅ Environment variables are set
- ✅ Supabase connection works
- ✅ All database tables exist
- ✅ All storage buckets exist
- ✅ Database operations work

### **4. Run the Server**

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API available at: `http://localhost:8000`  
Interactive docs: `http://localhost:8000/docs`  
Health check: `http://localhost:8000/health`

---

## 🔌 API Endpoints

### **Upload & Preprocessing**
```http
POST /api/v1/upload
```
- Uploads image
- Removes background (optional)
- Creates project in database
- Returns `project_id`

**Example:**
```bash
curl -X POST http://localhost:8000/api/v1/upload \
  -F "file=@image.jpg" \
  -F "remove_bg=true"
```

**Response:**
```json
{
  "project_id": "abc-123-def-456",
  "original_url": "https://.../project-uploads/abc-123/original.png",
  "processed_url": "https://.../processed-images/abc-123/processed.png"
}
```

---

### **3D Generation**
```http
POST /api/v1/generate/{project_id}
```
- Generates 16 multi-view images (SyncDreamer)
- Estimates depth maps (MiDaS)
- Creates initial 3D Gaussian Splat model

**Example:**
```bash
curl -X POST http://localhost:8000/api/v1/generate/abc-123
```

**Response:**
```json
{
  "status": "completed",
  "model_url": "https://.../3d-models/abc-123/models/v0.ply",
  "views_generated": 16,
  "depth_maps_generated": 16
}
```

---

### **Refinement**
```http
POST /api/v1/refine/{project_id}
```
- Iterative SDXL enhancement + MVCRM back-projection
- Tracks quality metrics (PSNR, SSIM, LPIPS)
- Auto-converges when improvement < 1%

**Request Body:**
```json
{
  "num_iterations": 3,
  "learning_rate": 0.01,
  "views_to_refine": [0, 1, 2, 3, 4, 5, 6, 7]  // Optional
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/api/v1/refine/abc-123 \
  -H "Content-Type: application/json" \
  -d '{"num_iterations": 3, "learning_rate": 0.01}'
```

**Response:**
```json
{
  "status": "refined",
  "iterations_completed": 3,
  "converged": true,
  "final_model_url": "https://.../3d-models/abc-123/models/v3.ply",
  "metrics": [
    {"iteration": 1, "overall_quality": 0.82, "psnr": 29.0},
    {"iteration": 2, "overall_quality": 0.86, "psnr": 29.5},
    {"iteration": 3, "overall_quality": 0.88, "psnr": 29.8}
  ]
}
```

---

### **Export**
```http
GET /api/v1/export/{project_id}?format=glb&optimize=true
```
- Converts model to requested format
- Supports: `.ply`, `.splat`, `.glb`, `.obj`, `.usdz`
- Optional file size optimization

**Example:**
```bash
curl "http://localhost:8000/api/v1/export/abc-123?format=glb&optimize=true"
```

**Response:**
```json
{
  "download_url": "https://.../3d-models/abc-123/models/final.glb",
  "format": "glb",
  "file_size_mb": 25.3
}
```

---

### **Real-Time Status**
```http
GET /api/v1/status/{project_id}
```
- Get current pipeline status
- View progress percentage
- Access all intermediate outputs

**Example:**
```bash
curl http://localhost:8000/api/v1/status/abc-123
```

**Response:**
```json
{
  "project": {
    "status": "refining",
    "current_step": "Iteration 2: Enhancing with SDXL",
    "progress_percentage": 82
  },
  "pipeline_progress": {
    "multiview_images": {"completed": 16, "total": 16},
    "depth_maps": {"completed": 16, "total": 16},
    "model_versions": 2,
    "refinement_iterations": 2
  },
  "iterations": [
    {"iteration_number": 1, "overall_quality": 0.82},
    {"iteration_number": 2, "overall_quality": 0.86}
  ]
}
```

---

## 🗄️ Database Integration

All pipeline data is stored in **Supabase**:

### **Tables**
- `projects` - Master project tracking
- `multiview_generation` - 16 views per project
- `depth_maps` - Depth estimations
- `gaussian_splat_models` - 3D model versions
- `enhancement_iterations` - Refinement tracking
- `enhanced_views` - SDXL outputs
- `refinement_metrics` - Quality metrics
- `export_history` - Export records

### **Storage Buckets**
- `project-uploads` - Original images
- `processed-images` - Background-removed
- `multiview-images` - 16 views
- `depth-maps` - Depth maps + heatmaps
- `enhanced-views` - SDXL outputs
- `3d-models` - .ply/.splat models

See **[PIPELINE_DIAGRAM.md](PIPELINE_DIAGRAM.md)** for visual data flow.

---

## 🧪 Testing

### **Run Integration Tests**
```bash
python test_integration.py
```

### **Manual API Testing**
Use the interactive Swagger UI: http://localhost:8000/docs

### **Test Upload**
```bash
curl -X POST http://localhost:8000/api/v1/upload \
  -F "file=@../assets/sample_inputs/chair.png"
```

---

## 🐳 Docker Deployment

```bash
docker build -t glimpse3d-backend .
docker run -p 8000:8000 --env-file .env glimpse3d-backend
```

See **[Dockerfile](Dockerfile)** for details.

---

## 📚 Documentation

- **[SUPABASE_SETUP.md](SUPABASE_SETUP.md)** - Complete Supabase setup guide
- **[BUCKET_POLICIES.md](BUCKET_POLICIES.md)** - Storage bucket configurations
- **[QUICKSTART.md](QUICKSTART.md)** - 5-minute quick start
- **[PIPELINE_DIAGRAM.md](PIPELINE_DIAGRAM.md)** - Visual pipeline flow
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Implementation details

---

## 🔧 Configuration

Edit `backend/.env`:

```bash
# Supabase
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_ANON_KEY=eyJhbGci...

# Model Paths
MODEL_DIR=model_checkpoints
ASSET_DIR=assets

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=true
```

See **[.env.example](.env.example)** for all options.

---

## 🛠️ Development

### **Project Structure**
- Routes in `app/routes/` define API endpoints
- Database operations in `app/core/database.py`
- Storage uploads in `app/core/storage.py`
- AI services in `app/services/`

### **Adding a New Route**
1. Create `app/routes/my_route.py`
2. Import in `app/main.py`
3. Register: `app.include_router(my_route.router, prefix="/api/v1")`

### **Database Helpers**
```python
from app.core.database import DatabaseManager

# Create project
project_id = DatabaseManager.create_project()

# Update status
DatabaseManager.update_project_status(project_id, "processing", "Current step")

# Save data
DatabaseManager.save_multiview_images(project_id, views_data)
```

### **Storage Helpers**
```python
from app.core.storage import StorageManager

# Upload image
url = StorageManager.upload_original_image(project_id, image)

# Upload numpy array
url = StorageManager.upload_depth_map(project_id, view_index, depth_array)
```

---

## 🐛 Troubleshooting

**"Missing Supabase credentials"**  
→ Create `.env` file with `SUPABASE_URL` and `SUPABASE_ANON_KEY`

**"Table does not exist"**  
→ Run `supabase_schema.sql` in Supabase SQL Editor

**"Bucket does not exist"**  
→ Create all 6 buckets in Supabase Storage (see [SUPABASE_SETUP.md](SUPABASE_SETUP.md))

**"Row-level security policy violation"**  
→ Apply bucket policies from [BUCKET_POLICIES.md](BUCKET_POLICIES.md)

---

## 📊 Monitoring

### **View Active Projects**
```sql
SELECT id, status, created_at, current_step 
FROM projects 
WHERE status NOT IN ('completed', 'failed')
ORDER BY created_at DESC;
```

### **View Refinement Progress**
```sql
SELECT * FROM refinement_progress 
WHERE project_id = '<project_id>';
```

### **Check Storage Usage**
```sql
SELECT 
    bucket_id,
    COUNT(*) as file_count,
    SUM((metadata->>'size')::bigint) / 1024 / 1024 as size_mb
FROM storage.objects
GROUP BY bucket_id;
```

---

## 🤝 Contributing

1. Follow existing code structure
2. Use `DatabaseManager` and `StorageManager` helpers
3. Update documentation for new features
4. Test with `test_integration.py`

---

## 📝 License

See main project [LICENSE](../LICENSE)

---

## 🙋 Support

- Detailed setup: [SUPABASE_SETUP.md](SUPABASE_SETUP.md)
- Quick start: [QUICKSTART.md](QUICKSTART.md)
- Pipeline flow: [PIPELINE_DIAGRAM.md](PIPELINE_DIAGRAM.md)
- API docs: http://localhost:8000/docs

---

**Built with FastAPI + PyTorch + Supabase** 🚀
- `POST /generate/{id}`: Trigger initial coarse 3D reconstruction.
- `POST /refine/{id}`: Run the **Enhancement Loop** (SDXL + Backprojection).
- `GET /export/{id}`: Download the final 3D model.
