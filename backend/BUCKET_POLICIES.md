# Storage Bucket Policies for Supabase

This file contains the storage bucket policies that need to be applied in Supabase.

## Overview

Since we want **public access without authentication**, we need to create permissive policies for all storage buckets.

---

## Option 1: SQL Policy Creation (Recommended)

Run this SQL script in Supabase SQL Editor after creating the buckets:

```sql
-- ================================================
-- STORAGE BUCKET POLICIES - PUBLIC ACCESS
-- ================================================

-- Policy for: project-uploads bucket
CREATE POLICY "Public read access for project-uploads"
ON storage.objects FOR SELECT
USING (bucket_id = 'project-uploads');

CREATE POLICY "Public insert access for project-uploads"
ON storage.objects FOR INSERT
WITH CHECK (bucket_id = 'project-uploads');

CREATE POLICY "Public update access for project-uploads"
ON storage.objects FOR UPDATE
USING (bucket_id = 'project-uploads');

-- Policy for: processed-images bucket
CREATE POLICY "Public read access for processed-images"
ON storage.objects FOR SELECT
USING (bucket_id = 'processed-images');

CREATE POLICY "Public insert access for processed-images"
ON storage.objects FOR INSERT
WITH CHECK (bucket_id = 'processed-images');

CREATE POLICY "Public update access for processed-images"
ON storage.objects FOR UPDATE
USING (bucket_id = 'processed-images');

-- Policy for: multiview-images bucket
CREATE POLICY "Public read access for multiview-images"
ON storage.objects FOR SELECT
USING (bucket_id = 'multiview-images');

CREATE POLICY "Public insert access for multiview-images"
ON storage.objects FOR INSERT
WITH CHECK (bucket_id = 'multiview-images');

CREATE POLICY "Public update access for multiview-images"
ON storage.objects FOR UPDATE
USING (bucket_id = 'multiview-images');

-- Policy for: depth-maps bucket
CREATE POLICY "Public read access for depth-maps"
ON storage.objects FOR SELECT
USING (bucket_id = 'depth-maps');

CREATE POLICY "Public insert access for depth-maps"
ON storage.objects FOR INSERT
WITH CHECK (bucket_id = 'depth-maps');

CREATE POLICY "Public update access for depth-maps"
ON storage.objects FOR UPDATE
USING (bucket_id = 'depth-maps');

-- Policy for: enhanced-views bucket
CREATE POLICY "Public read access for enhanced-views"
ON storage.objects FOR SELECT
USING (bucket_id = 'enhanced-views');

CREATE POLICY "Public insert access for enhanced-views"
ON storage.objects FOR INSERT
WITH CHECK (bucket_id = 'enhanced-views');

CREATE POLICY "Public update access for enhanced-views"
ON storage.objects FOR UPDATE
USING (bucket_id = 'enhanced-views');

-- Policy for: 3d-models bucket
CREATE POLICY "Public read access for 3d-models"
ON storage.objects FOR SELECT
USING (bucket_id = '3d-models');

CREATE POLICY "Public insert access for 3d-models"
ON storage.objects FOR INSERT
WITH CHECK (bucket_id = '3d-models');

CREATE POLICY "Public update access for 3d-models"
ON storage.objects FOR UPDATE
USING (bucket_id = '3d-models');
```

---

## Option 2: UI Policy Creation

For each bucket, follow these steps in the Supabase Dashboard:

### 1. Navigate to Storage Policies

1. Go to **Storage** in the left sidebar
2. Click on the bucket name (e.g., `project-uploads`)
3. Click the **"Policies"** tab
4. Click **"New Policy"**

### 2. Create Read Policy

- **Policy name**: `Public read access`
- **Allowed operation**: `SELECT` (read)
- **Policy definition**: `true`
- **Target roles**: `public` (or leave default)
- Click **"Review"** → **"Save policy"**

### 3. Create Write Policy

- **Policy name**: `Public write access`
- **Allowed operation**: `INSERT` (create)
- **Policy definition**: `true`
- **Target roles**: `public`
- Click **"Review"** → **"Save policy"**

### 4. Create Update Policy

- **Policy name**: `Public update access`
- **Allowed operation**: `UPDATE` (modify)
- **Policy definition**: `true`
- **Target roles**: `public`
- Click **"Review"** → **"Save policy"**

### 5. Repeat for All Buckets

Apply the above 3 policies to each of these buckets:
- ✅ `project-uploads`
- ✅ `processed-images`
- ✅ `multiview-images`
- ✅ `depth-maps`
- ✅ `enhanced-views`
- ✅ `3d-models`

---

## Option 3: Alternative - Make Buckets Public

If you want maximum simplicity (less secure but easier):

1. When creating each bucket, toggle **"Public bucket"** to **ON**
2. This automatically allows public read access
3. You still need to add INSERT/UPDATE policies via SQL or UI

---

## Verifying Policies

Run this query to check if policies exist:

```sql
SELECT 
    schemaname,
    tablename,
    policyname,
    permissive,
    roles,
    cmd
FROM pg_policies
WHERE tablename = 'objects'
ORDER BY policyname;
```

You should see policies for all 6 buckets with operations: `SELECT`, `INSERT`, `UPDATE`.

---

## Security Notes

### ⚠️ Public Access Risks

Since these buckets are fully public:
- Anyone with the URL can view files
- Anyone can upload files (rate-limited by Supabase)
- No user authentication required

### 🛡️ Recommended Protections

1. **Add path restrictions** (more secure option):

```sql
-- Only allow uploads to paths matching: {project_id}/*
CREATE POLICY "Restrict upload paths for project-uploads"
ON storage.objects FOR INSERT
WITH CHECK (
    bucket_id = 'project-uploads' 
    AND (storage.foldername(name))[1] ~ '^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$'
);
```

This ensures files can only be uploaded to valid UUID folders.

2. **Add file size limits**:

In your FastAPI backend:
```python
from fastapi import File, UploadFile

@router.post("/upload")
async def upload_image(
    file: UploadFile = File(..., max_length=10_000_000)  # 10 MB limit
):
    ...
```

3. **Add content-type restrictions**:

```python
ALLOWED_TYPES = ["image/png", "image/jpeg", "image/jpg"]

if file.content_type not in ALLOWED_TYPES:
    raise HTTPException(400, "Invalid file type")
```

4. **Monitor storage usage**:

```sql
-- Get total storage used per bucket
SELECT 
    bucket_id,
    COUNT(*) as file_count,
    SUM(metadata->>'size')::bigint / 1024 / 1024 as size_mb
FROM storage.objects
GROUP BY bucket_id;
```

---

## Cleanup Policies (Optional)

To automatically delete files older than 30 days:

```sql
-- Create a function to clean up old files
CREATE OR REPLACE FUNCTION cleanup_old_storage()
RETURNS void AS $$
BEGIN
    -- Delete storage objects for projects older than 30 days
    DELETE FROM storage.objects
    WHERE bucket_id IN (
        'project-uploads', 'processed-images', 'multiview-images',
        'depth-maps', 'enhanced-views', '3d-models'
    )
    AND created_at < NOW() - INTERVAL '30 days';
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;
```

Then set up a cron job (requires Supabase Pro):

```sql
-- Run cleanup weekly (requires pg_cron extension)
SELECT cron.schedule(
    'cleanup-old-storage',
    '0 2 * * 0',  -- Every Sunday at 2 AM
    'SELECT cleanup_old_storage();'
);
```

---

## Testing Bucket Access

### Test Upload via Python

```python
from app.core.storage import StorageManager
from PIL import Image
import numpy as np

# Create a test image
test_image = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))

# Upload to project-uploads bucket
url = StorageManager.upload_original_image("test-project-id", test_image)
print(f"✅ Uploaded to: {url}")

# Verify the URL is publicly accessible
import requests
response = requests.get(url)
print(f"✅ Public access: {response.status_code == 200}")
```

### Test via cURL

```bash
# Upload a file
curl -X POST \
  'https://xxxxx.supabase.co/storage/v1/object/project-uploads/test/image.png' \
  -H 'Authorization: Bearer YOUR_ANON_KEY' \
  -H 'Content-Type: image/png' \
  --data-binary '@test.png'

# Download (public access)
curl 'https://xxxxx.supabase.co/storage/v1/object/public/project-uploads/test/image.png' \
  --output downloaded.png
```

---

## Summary

✅ **What you need to do:**
1. Create 6 storage buckets (all public)
2. Apply policies via SQL or UI (3 policies per bucket = 18 total)
3. Verify policies with the SQL query above
4. Test upload/download with Python or cURL

✅ **What this enables:**
- Frontend can directly upload images to buckets
- Anyone can view generated 3D models via public URLs
- No authentication needed (good for demos/research)
- Supabase handles all storage management

⚠️ **Don't forget:**
- Add rate limiting in your FastAPI app
- Monitor storage usage in Supabase dashboard
- Set up cleanup jobs for old projects
- Consider adding more restrictive policies for production

---

Need help? Check the [Supabase Storage documentation](https://supabase.com/docs/guides/storage).
