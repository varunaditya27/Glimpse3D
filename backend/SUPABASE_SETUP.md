# Supabase Setup Instructions for Glimpse3D

This guide walks you through setting up your Supabase database and storage for the Glimpse3D project.

---

## 📋 Prerequisites

- Supabase account (free tier works fine): https://supabase.com
- Project created in Supabase dashboard

---

## 🚀 Step-by-Step Setup

### **Step 1: Create a New Supabase Project**

1. Go to https://app.supabase.com
2. Click **"New Project"**
3. Fill in:
   - **Name**: `Glimpse3D` (or your preferred name)
   - **Database Password**: Generate a secure password (save this!)
   - **Region**: Choose closest to your users
   - **Pricing Plan**: Free tier is sufficient for development
4. Click **"Create new project"** and wait ~2 minutes for setup

---

### **Step 2: Get Your API Credentials**

1. In your Supabase project dashboard, go to **Settings** → **API**
2. Copy the following values:
   - **Project URL** (e.g., `https://xxxxx.supabase.co`)
   - **anon public** key (long string starting with `eyJ...`)

3. Create a `.env` file in your `backend/` directory:

```bash
# backend/.env
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

⚠️ **Important**: Add `.env` to your `.gitignore` to prevent committing credentials!

---

### **Step 3: Run the Database Schema**

1. In Supabase dashboard, go to **SQL Editor** (left sidebar)
2. Click **"New query"**
3. Copy the entire contents of `backend/supabase_schema.sql`
4. Paste into the SQL editor
5. Click **"Run"** (or press `Ctrl+Enter`)

You should see a success message: ✅ **"Success. No rows returned"**

**What this creates:**
- 8 tables: `projects`, `multiview_generation`, `depth_maps`, `gaussian_splat_models`, `enhancement_iterations`, `enhanced_views`, `refinement_metrics`, `export_history`
- Indexes for fast queries
- Row-level security policies (public access enabled)
- Helper views for analytics

---

### **Step 4: Create Storage Buckets**

You need to create 6 storage buckets for different file types.

#### **Option A: Manual Creation (Recommended for first-time setup)**

For each bucket below:

1. Go to **Storage** (left sidebar)
2. Click **"Create a new bucket"**
3. Enter the bucket name
4. Set **Public bucket** to ✅ **ON** (important!)
5. Click **"Create bucket"**

**Buckets to create:**

| Bucket Name | Description | Public? |
|------------|-------------|---------|
| `project-uploads` | Original user images | ✅ Yes |
| `processed-images` | Background-removed images | ✅ Yes |
| `multiview-images` | SyncDreamer 16-view outputs | ✅ Yes |
| `depth-maps` | MiDaS depth maps & heatmaps | ✅ Yes |
| `enhanced-views` | SDXL-enhanced images | ✅ Yes |
| `3d-models` | Gaussian Splat models & exports | ✅ Yes |

#### **Option B: Programmatic Creation (Advanced)**

You can also create buckets via SQL:

```sql
-- Run this in the SQL Editor
INSERT INTO storage.buckets (id, name, public) VALUES
    ('project-uploads', 'project-uploads', true),
    ('processed-images', 'processed-images', true),
    ('multiview-images', 'multiview-images', true),
    ('depth-maps', 'depth-maps', true),
    ('enhanced-views', 'enhanced-views', true),
    ('3d-models', '3d-models', true);
```

---

### **Step 5: Configure Bucket Policies**

Since we want public access (no authentication), we need to set permissive policies.

1. Go to **Storage** → Click on a bucket (e.g., `project-uploads`)
2. Click **"Policies"** tab
3. Click **"New Policy"**
4. Select **"For full customization"**
5. Use these settings:

**Policy Name**: `Public Read and Write`

**Target Roles**: `public` (select from dropdown)

**Policy Definition**:

For **SELECT** (read access):
```sql
true
```

For **INSERT** (upload access):
```sql
true
```

For **UPDATE** (modify access):
```sql
true
```

For **DELETE** (delete access) - Optional:
```sql
true
```

6. Click **"Review"** → **"Save policy"**

**Repeat for all 6 buckets**.

---

### **Step 6: Set Storage Limits (Optional)**

To prevent abuse, you can set size limits:

1. Go to **Settings** → **Storage**
2. Set **Maximum file size**: `100 MB` (adjust as needed)
3. Set **Total storage limit**: `5 GB` (free tier limit)

---

### **Step 7: Verify Setup**

Run this SQL query to check everything is created:

```sql
-- Check tables exist
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public' 
ORDER BY table_name;

-- Check buckets exist
SELECT * FROM storage.buckets;

-- Expected output: 8 tables + 6 buckets
```

You should see:
- **Tables**: `depth_maps`, `enhancement_iterations`, `enhanced_views`, `export_history`, `gaussian_splat_models`, `multiview_generation`, `projects`, `refinement_metrics`
- **Buckets**: All 6 storage buckets listed

---

### **Step 8: Test Database Connection**

In your backend directory, run:

```bash
cd backend
pip install supabase
python -c "from app.core.supabase_client import get_supabase; client = get_supabase(); print('✅ Connection successful!')"
```

If you see `✅ Connection successful!`, you're all set!

---

## 🔒 Security Considerations

Since this is **public access** (no authentication):

### ✅ **What's Protected:**
- Your `SUPABASE_ANON_KEY` is safe to expose (it's meant for public use)
- Database queries are still logged and rate-limited by Supabase
- You have Row Level Security (RLS) enabled with public policies

### ⚠️ **What to Monitor:**
- **Storage usage**: Free tier = 1 GB (monitor in dashboard)
- **Database rows**: Free tier = unlimited rows, but watch query performance
- **Bandwidth**: Free tier = 2 GB/month egress

### 🛡️ **Recommended Protections:**

1. **Rate Limiting**: Add rate limiting in FastAPI:
```python
from slowapi import Limiter
limiter = Limiter(key_func=lambda: "global")

@app.post("/upload")
@limiter.limit("10/minute")  # Max 10 uploads per minute
async def upload_image(...):
    ...
```

2. **File Size Limits**: Already configured in FastAPI:
```python
from fastapi import File, UploadFile
@app.post("/upload")
async def upload_image(file: UploadFile = File(..., max_length=10_000_000)):  # 10 MB
    ...
```

3. **CORS**: Restrict which domains can access your API:
```python
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Only your frontend
    allow_methods=["POST", "GET"],
)
```

4. **Cleanup Job**: Add a scheduled task to delete old projects (>30 days):
```sql
-- Run this weekly as a cron job
DELETE FROM projects WHERE created_at < NOW() - INTERVAL '30 days';
```

---

## 📊 Monitoring & Analytics

### **View Project Statistics**

Use the built-in views:

```sql
-- Get project summary
SELECT * FROM project_summary WHERE id = '<project_id>';

-- Get refinement progress
SELECT * FROM refinement_progress WHERE project_id = '<project_id>';

-- Get all active projects
SELECT id, status, created_at, current_step 
FROM projects 
WHERE status NOT IN ('completed', 'failed')
ORDER BY created_at DESC;

-- Get storage usage per project
SELECT 
    p.id,
    COUNT(DISTINCT mv.id) as num_views,
    COUNT(DISTINCT dm.id) as num_depth_maps,
    COUNT(DISTINCT ev.id) as num_enhanced_views
FROM projects p
LEFT JOIN multiview_generation mv ON p.id = mv.project_id
LEFT JOIN depth_maps dm ON p.id = dm.project_id
LEFT JOIN enhanced_views ev ON ev.iteration_id IN (
    SELECT id FROM enhancement_iterations WHERE project_id = p.id
)
GROUP BY p.id;
```

---

## 🧪 Testing the Integration

### **1. Test Image Upload**

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@test_image.jpg" \
  -F "remove_bg=true"
```

Expected response:
```json
{
  "project_id": "uuid-here",
  "status": "uploaded",
  "original_url": "https://...supabase.co/storage/v1/object/public/project-uploads/...",
  "processed_url": "https://...supabase.co/storage/v1/object/public/processed-images/..."
}
```

### **2. Check Database Entry**

```sql
SELECT * FROM projects ORDER BY created_at DESC LIMIT 1;
```

You should see your newly created project!

---

## 🆘 Troubleshooting

### **Error: "Missing Supabase credentials"**
✅ **Solution**: Make sure `.env` file exists in `backend/` directory with correct values

### **Error: "Bucket does not exist"**
✅ **Solution**: Verify all 6 buckets are created and names match exactly (case-sensitive)

### **Error: "new row violates row-level security policy"**
✅ **Solution**: Make sure you ran the bucket policies setup (Step 5) for all buckets

### **Error: "Storage quota exceeded"**
✅ **Solution**: 
- Delete old projects: `DELETE FROM projects WHERE created_at < NOW() - INTERVAL '7 days';`
- Upgrade to Supabase Pro plan ($25/month for 100 GB)

### **Uploads are slow**
✅ **Solution**: Choose a Supabase region closer to your users when creating the project

---

## 🎯 Next Steps

Once setup is complete:

1. ✅ Start your backend: `cd backend && uvicorn app.main:app --reload`
2. ✅ Test upload endpoint
3. ✅ Integrate your AI modules (SyncDreamer, MiDaS, gsplat)
4. ✅ Replace placeholder code with actual model inference
5. ✅ Build your frontend to display project status in real-time

---

## 📚 Additional Resources

- **Supabase Docs**: https://supabase.com/docs
- **Supabase Python Client**: https://github.com/supabase-community/supabase-py
- **Storage API**: https://supabase.com/docs/guides/storage
- **RLS Policies**: https://supabase.com/docs/guides/auth/row-level-security

---

## 🎉 You're All Set!

Your Glimpse3D backend is now configured with:
- ✅ 8 database tables for tracking the entire pipeline
- ✅ 6 storage buckets for all file types
- ✅ Public access policies (no authentication required)
- ✅ Analytics views for monitoring
- ✅ Python helper functions for easy integration

Start building! 🚀
