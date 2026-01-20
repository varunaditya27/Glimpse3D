-- ================================================
-- Database Migration: Add Image Validation Support
-- ================================================
-- Run this if you already have the database schema set up
-- and need to add the validation_metadata column
-- ================================================

-- Add validation_metadata column to projects table
ALTER TABLE projects 
ADD COLUMN IF NOT EXISTS validation_metadata JSONB;

-- Add comment explaining the column
COMMENT ON COLUMN projects.validation_metadata IS 
'Image validation metrics including blur score, object count, dimensions, etc.';

-- Example validation_metadata structure:
-- {
--   "original_size": [1024, 768],
--   "processed_size": [512, 512],
--   "num_objects_detected": 1,
--   "object_area_pixels": 45678,
--   "blur_score": 234.5,
--   "validation_passed": true
-- }

-- Verify the change
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'projects' 
AND column_name = 'validation_metadata';
