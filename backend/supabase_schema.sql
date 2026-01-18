-- ================================================
-- Glimpse3D Supabase Database Schema
-- ================================================
-- This schema tracks the entire 3D generation pipeline
-- from image upload to final model export.
-- ================================================

-- Enable UUID extension for primary keys
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ================================================
-- TABLE 1: projects
-- Tracks each 3D generation project/session
-- ================================================
CREATE TABLE projects (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    status TEXT NOT NULL DEFAULT 'uploading' CHECK (status IN (
        'uploading',
        'preprocessing',
        'multiview_generating',
        'depth_estimating',
        'reconstructing',
        'enhancing',
        'refining',
        'completed',
        'failed'
    )),
    current_step TEXT,
    error_message TEXT,
    original_image_url TEXT,
    processed_image_url TEXT,
    final_model_url TEXT,
    total_processing_time FLOAT DEFAULT 0,
    user_session_id TEXT -- Optional: for tracking across browser sessions
);

-- Index for faster queries
CREATE INDEX idx_projects_status ON projects(status);
CREATE INDEX idx_projects_created_at ON projects(created_at);

-- ================================================
-- TABLE 2: multiview_generation
-- Stores SyncDreamer output (16 views)
-- ================================================
CREATE TABLE multiview_generation (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    view_index INTEGER NOT NULL CHECK (view_index >= 0 AND view_index < 16),
    elevation FLOAT,
    azimuth FLOAT,
    image_url TEXT NOT NULL,
    generated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(project_id, view_index)
);

CREATE INDEX idx_multiview_project ON multiview_generation(project_id);

-- ================================================
-- TABLE 3: depth_maps
-- Stores MiDaS depth estimation results
-- ================================================
CREATE TABLE depth_maps (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    view_index INTEGER NOT NULL CHECK (view_index >= 0 AND view_index < 16),
    depth_map_url TEXT NOT NULL, -- .npy file URL
    depth_heatmap_url TEXT, -- Colored visualization PNG
    min_depth FLOAT,
    max_depth FLOAT,
    mean_depth FLOAT,
    confidence_score FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(project_id, view_index)
);

CREATE INDEX idx_depth_maps_project ON depth_maps(project_id);

-- ================================================
-- TABLE 4: gaussian_splat_models
-- Stores 3D Gaussian Splat model versions
-- ================================================
CREATE TABLE gaussian_splat_models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    version INTEGER NOT NULL DEFAULT 0,
    model_file_url TEXT NOT NULL, -- .ply or .splat file
    num_splats INTEGER,
    file_size_mb FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_final BOOLEAN DEFAULT FALSE,
    UNIQUE(project_id, version)
);

CREATE INDEX idx_gs_models_project ON gaussian_splat_models(project_id);

-- ================================================
-- TABLE 5: enhancement_iterations
-- Tracks SDXL + MVCRM refinement iterations
-- ================================================
CREATE TABLE enhancement_iterations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    iteration_number INTEGER NOT NULL,
    learning_rate FLOAT,
    views_processed INTEGER DEFAULT 0,
    avg_depth_consistency FLOAT,
    avg_feature_similarity FLOAT,
    psnr FLOAT,
    ssim FLOAT,
    lpips FLOAT,
    overall_quality FLOAT,
    converged BOOLEAN DEFAULT FALSE,
    processing_time FLOAT, -- Time taken for this iteration (seconds)
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(project_id, iteration_number)
);

CREATE INDEX idx_iterations_project ON enhancement_iterations(project_id);

-- ================================================
-- TABLE 6: enhanced_views
-- Stores SDXL-enhanced images for each iteration
-- ================================================
CREATE TABLE enhanced_views (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    iteration_id UUID NOT NULL REFERENCES enhancement_iterations(id) ON DELETE CASCADE,
    view_index INTEGER NOT NULL,
    rendered_image_url TEXT, -- Pre-enhancement render from current model
    enhanced_image_url TEXT NOT NULL, -- Post-SDXL enhancement
    prompt_used TEXT,
    negative_prompt TEXT,
    controlnet_scale FLOAT,
    guidance_scale FLOAT,
    num_inference_steps INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(iteration_id, view_index)
);

CREATE INDEX idx_enhanced_views_iteration ON enhanced_views(iteration_id);

-- ================================================
-- TABLE 7: refinement_metrics
-- Detailed per-metric tracking for analysis
-- ================================================
CREATE TABLE refinement_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    iteration_id UUID NOT NULL REFERENCES enhancement_iterations(id) ON DELETE CASCADE,
    metric_name TEXT NOT NULL, -- e.g., 'psnr', 'depth_variance', 'clip_similarity'
    metric_value FLOAT NOT NULL,
    improvement_over_baseline FLOAT,
    view_index INTEGER, -- If metric is per-view, otherwise NULL for global metrics
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_metrics_iteration ON refinement_metrics(iteration_id);
CREATE INDEX idx_metrics_name ON refinement_metrics(metric_name);

-- ================================================
-- TABLE 8: export_history
-- Tracks exported models in different formats
-- ================================================
CREATE TABLE export_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    format TEXT NOT NULL CHECK (format IN ('ply', 'splat', 'glb', 'obj', 'usdz')),
    file_url TEXT NOT NULL,
    file_size_mb FLOAT,
    optimization_level TEXT, -- 'none', 'medium', 'high'
    exported_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_export_project ON export_history(project_id);

-- ================================================
-- ROW LEVEL SECURITY (RLS) - DISABLED FOR PUBLIC ACCESS
-- ================================================
-- Since you want no authentication, we'll disable RLS
-- and make tables publicly accessible via anon key

ALTER TABLE projects ENABLE ROW LEVEL SECURITY;
ALTER TABLE multiview_generation ENABLE ROW LEVEL SECURITY;
ALTER TABLE depth_maps ENABLE ROW LEVEL SECURITY;
ALTER TABLE gaussian_splat_models ENABLE ROW LEVEL SECURITY;
ALTER TABLE enhancement_iterations ENABLE ROW LEVEL SECURITY;
ALTER TABLE enhanced_views ENABLE ROW LEVEL SECURITY;
ALTER TABLE refinement_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE export_history ENABLE ROW LEVEL SECURITY;

-- Create public access policies (allow all operations with anon key)
CREATE POLICY "Public read access" ON projects FOR SELECT USING (true);
CREATE POLICY "Public insert access" ON projects FOR INSERT WITH CHECK (true);
CREATE POLICY "Public update access" ON projects FOR UPDATE USING (true);

CREATE POLICY "Public read access" ON multiview_generation FOR SELECT USING (true);
CREATE POLICY "Public insert access" ON multiview_generation FOR INSERT WITH CHECK (true);

CREATE POLICY "Public read access" ON depth_maps FOR SELECT USING (true);
CREATE POLICY "Public insert access" ON depth_maps FOR INSERT WITH CHECK (true);

CREATE POLICY "Public read access" ON gaussian_splat_models FOR SELECT USING (true);
CREATE POLICY "Public insert access" ON gaussian_splat_models FOR INSERT WITH CHECK (true);

CREATE POLICY "Public read access" ON enhancement_iterations FOR SELECT USING (true);
CREATE POLICY "Public insert access" ON enhancement_iterations FOR INSERT WITH CHECK (true);

CREATE POLICY "Public read access" ON enhanced_views FOR SELECT USING (true);
CREATE POLICY "Public insert access" ON enhanced_views FOR INSERT WITH CHECK (true);

CREATE POLICY "Public read access" ON refinement_metrics FOR SELECT USING (true);
CREATE POLICY "Public insert access" ON refinement_metrics FOR INSERT WITH CHECK (true);

CREATE POLICY "Public read access" ON export_history FOR SELECT USING (true);
CREATE POLICY "Public insert access" ON export_history FOR INSERT WITH CHECK (true);

-- ================================================
-- FUNCTIONS & TRIGGERS
-- ================================================

-- Auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_projects_updated_at
    BEFORE UPDATE ON projects
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ================================================
-- VIEWS FOR ANALYTICS
-- ================================================

-- View to get project summary with all related data counts
CREATE VIEW project_summary AS
SELECT 
    p.id,
    p.status,
    p.created_at,
    p.total_processing_time,
    COUNT(DISTINCT mv.id) AS num_views_generated,
    COUNT(DISTINCT dm.id) AS num_depth_maps,
    COUNT(DISTINCT gs.id) AS num_model_versions,
    COUNT(DISTINCT ei.id) AS num_refinement_iterations,
    MAX(ei.overall_quality) AS best_quality_score
FROM projects p
LEFT JOIN multiview_generation mv ON p.id = mv.project_id
LEFT JOIN depth_maps dm ON p.id = dm.project_id
LEFT JOIN gaussian_splat_models gs ON p.id = gs.project_id
LEFT JOIN enhancement_iterations ei ON p.id = ei.project_id
GROUP BY p.id;

-- View for tracking refinement progress
CREATE VIEW refinement_progress AS
SELECT 
    ei.project_id,
    ei.iteration_number,
    ei.overall_quality,
    ei.psnr,
    ei.ssim,
    ei.converged,
    ei.processing_time,
    COUNT(ev.id) AS num_views_enhanced
FROM enhancement_iterations ei
LEFT JOIN enhanced_views ev ON ei.id = ev.iteration_id
GROUP BY ei.id, ei.project_id, ei.iteration_number, ei.overall_quality, 
         ei.psnr, ei.ssim, ei.converged, ei.processing_time
ORDER BY ei.project_id, ei.iteration_number;

-- ================================================
-- SAMPLE QUERIES FOR TESTING
-- ================================================

-- Get all data for a specific project
-- SELECT * FROM projects WHERE id = '<project_id>';
-- SELECT * FROM multiview_generation WHERE project_id = '<project_id>';
-- SELECT * FROM depth_maps WHERE project_id = '<project_id>';
-- SELECT * FROM gaussian_splat_models WHERE project_id = '<project_id>' ORDER BY version;
-- SELECT * FROM enhancement_iterations WHERE project_id = '<project_id>' ORDER BY iteration_number;

-- Get refinement metrics for a project
-- SELECT ei.iteration_number, rm.metric_name, rm.metric_value 
-- FROM refinement_metrics rm
-- JOIN enhancement_iterations ei ON rm.iteration_id = ei.id
-- WHERE ei.project_id = '<project_id>'
-- ORDER BY ei.iteration_number, rm.metric_name;

-- Get final model URL
-- SELECT final_model_url FROM projects WHERE id = '<project_id>';
