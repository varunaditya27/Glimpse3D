# Quick Start Guide - Glimpse3D Supabase Integration

## 🚀 5-Minute Setup

### 1. Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. Set Up Supabase

**Create Project:**
- Go to https://app.supabase.com
- Create new project (takes ~2 minutes)
- Note your **Project URL** and **anon key**

**Create .env file:**
```bash
cd backend
cp .env.example .env
# Edit .env and add your Supabase credentials
```

**Run Database Schema:**
1. Go to Supabase Dashboard → SQL Editor
2. Copy `backend/supabase_schema.sql` → Paste → Run

**Create Storage Buckets:**
1. Go to Storage → New Bucket
2. Create these 6 buckets (all public):
   - `project-uploads`
   - `processed-images`
   - `multiview-images`
   - `depth-maps`
   - `enhanced-views`
   - `3d-models`

### 3. Test Connection
```bash
python -c "from app.core.supabase_client import get_supabase; get_supabase(); print('✅ Connected!')"
```

### 4. Start Backend
```bash
uvicorn app.main:app --reload
```

### 5. Test Upload
```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@test_image.jpg"
```

---

## 📊 Data Flow Overview

```
User Upload
    ↓
📦 project-uploads bucket
    ↓
🗄️ projects table (status: uploading)
    ↓
Background Removal
    ↓
📦 processed-images bucket
    ↓
🗄️ projects table (status: preprocessing)
    ↓
SyncDreamer (16 views)
    ↓
📦 multiview-images bucket (16 files)
    ↓
🗄️ multiview_generation table (16 rows)
    ↓
MiDaS Depth
    ↓
📦 depth-maps bucket (32 files: .npy + heatmaps)
    ↓
🗄️ depth_maps table (16 rows)
    ↓
Gaussian Splatting
    ↓
📦 3d-models bucket (v0.ply)
    ↓
🗄️ gaussian_splat_models table (v0)
    ↓
SDXL + MVCRM Refinement (iterative)
    ↓
📦 enhanced-views bucket (per iteration)
    ↓
🗄️ enhancement_iterations table
🗄️ enhanced_views table
🗄️ refinement_metrics table
    ↓
Final Model
    ↓
🗄️ projects table (status: completed)
```

---

## 🔌 API Endpoints

### POST /upload
Upload image → Create project → Return project_id

### POST /generate/{project_id}
Multi-view generation → Depth estimation → Initial 3D model

### POST /refine/{project_id}
Iterative SDXL enhancement → MVCRM back-projection

### GET /export/{project_id}?format=ply
Export final model in requested format

### GET /status/{project_id}
Get current pipeline status and progress

---

## 🗄️ Database Tables

| Table | Purpose |
|-------|---------|
| `projects` | Master project tracking |
| `multiview_generation` | SyncDreamer 16 views |
| `depth_maps` | MiDaS depth outputs |
| `gaussian_splat_models` | 3D model versions |
| `enhancement_iterations` | Refinement loop tracking |
| `enhanced_views` | SDXL-enhanced images |
| `refinement_metrics` | Quality metrics |
| `export_history` | Export records |

---

## 📦 Storage Buckets

| Bucket | Contains | Public? |
|--------|----------|---------|
| `project-uploads` | Original images | ✅ |
| `processed-images` | BG-removed | ✅ |
| `multiview-images` | 16 views | ✅ |
| `depth-maps` | Depth .npy + heatmaps | ✅ |
| `enhanced-views` | SDXL outputs | ✅ |
| `3d-models` | .ply/.splat models | ✅ |

---

## 🐍 Python Usage Examples

### Create a Project
```python
from app.core.database import DatabaseManager

project_id = DatabaseManager.create_project()
```

### Upload an Image
```python
from app.core.storage import StorageManager
from PIL import Image

image = Image.open("test.jpg")
url = StorageManager.upload_original_image(project_id, image)
```

### Save Multi-View Images
```python
views_data = [
    {"view_index": i, "elevation": 0, "azimuth": i*22.5, "image_url": f"url_{i}"}
    for i in range(16)
]
DatabaseManager.save_multiview_images(project_id, views_data)
```

### Track Refinement Iteration
```python
iteration_id = DatabaseManager.create_enhancement_iteration(
    project_id, iteration_number=1, learning_rate=0.01
)

DatabaseManager.update_iteration_metrics(
    iteration_id,
    psnr=28.5,
    ssim=0.85,
    overall_quality=0.90
)
```

---

## 🔍 Monitoring Queries

### Get Project Status
```sql
SELECT id, status, current_step, created_at 
FROM projects 
WHERE id = '<project_id>';
```

### Get All Multi-View Images
```sql
SELECT view_index, azimuth, image_url 
FROM multiview_generation 
WHERE project_id = '<project_id>' 
ORDER BY view_index;
```

### Get Refinement Progress
```sql
SELECT * FROM refinement_progress 
WHERE project_id = '<project_id>';
```

### Get Latest Model Version
```sql
SELECT * FROM gaussian_splat_models 
WHERE project_id = '<project_id>' 
ORDER BY version DESC 
LIMIT 1;
```

---

## 🐛 Common Issues

**"Missing Supabase credentials"**
→ Create `.env` file with SUPABASE_URL and SUPABASE_ANON_KEY

**"Bucket does not exist"**
→ Create all 6 buckets in Supabase Storage

**"Row-level security policy violation"**
→ Run the SQL policies from `BUCKET_POLICIES.md`

**Uploads fail silently**
→ Check bucket names match exactly (case-sensitive)

---

## 📚 Documentation Files

- **`SUPABASE_SETUP.md`** - Detailed setup guide
- **`BUCKET_POLICIES.md`** - Storage policy configurations
- **`supabase_schema.sql`** - Database schema
- **`.env.example`** - Environment variable template

---

## 🎯 Next Steps

1. ✅ Integrate SyncDreamer inference in [generate.py](backend/app/routes/generate.py#L49)
2. ✅ Integrate MiDaS depth in [generate.py](backend/app/routes/generate.py#L70)
3. ✅ Integrate gsplat reconstruction in [generate.py](backend/app/routes/generate.py#L99)
4. ✅ Integrate SDXL enhancement in [refine.py](backend/app/routes/refine.py#L88)
5. ✅ Integrate MVCRM back-projection in [refine.py](backend/app/routes/refine.py#L122)
6. ✅ Build frontend real-time status viewer
7. ✅ Add cleanup job for old projects

---

## 🙋 Need Help?

- Full setup instructions: `backend/SUPABASE_SETUP.md`
- Bucket policies: `backend/BUCKET_POLICIES.md`
- Supabase docs: https://supabase.com/docs
- Python client: https://github.com/supabase-community/supabase-py

---

**Ready to build!** 🚀
