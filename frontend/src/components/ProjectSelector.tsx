import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Image as ImageIcon, Calendar, Zap } from 'lucide-react';
import { Button } from './Button';

interface Project {
    id: string;
    original_image_url: string | null;
    processed_image_url: string | null;
    latest_model_url: string | null;
    status: string;
    created_at: string;
    total_processing_time?: number;
    model_info?: {
        version?: number;
        num_splats?: number;
        file_size_mb?: number;
    } | null;
    quality_metrics?: {
        overall_quality?: number;
        psnr?: number;
        ssim?: number;
    } | null;
}

interface ProjectSelectorProps {
    isOpen: boolean;
    onClose: () => void;
    onSelect: (project: Project) => void;
    selectedProjectId?: string | null;
    title?: string;
    apiUrl?: string;
}

export const ProjectSelector = ({
    isOpen,
    onClose,
    onSelect,
    selectedProjectId,
    title = "Select a Project",
    apiUrl = "http://localhost:8000/api/v1/compare/projects"
}: ProjectSelectorProps) => {
    const [projects, setProjects] = useState<Project[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        if (isOpen) {
            fetchProjects();
        }
    }, [isOpen]);

    const fetchProjects = async () => {
        setLoading(true);
        setError(null);
        
        try {
            const response = await fetch(apiUrl);
            if (!response.ok) {
                throw new Error('Failed to fetch projects');
            }
            
            const data = await response.json();
            setProjects(data.projects || []);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to load projects');
        } finally {
            setLoading(false);
        }
    };

    const formatDate = (dateString: string) => {
        const date = new Date(dateString);
        return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
    };

    const formatTime = (seconds?: number) => {
        if (!seconds) return 'N/A';
        if (seconds < 60) return `${Math.round(seconds)}s`;
        return `${Math.round(seconds / 60)}m`;
    };

    if (!isOpen) return null;

    return (
        <AnimatePresence>
            <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                style={{
                    position: 'fixed',
                    inset: 0,
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    backdropFilter: 'blur(5px)',
                    zIndex: 1000,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    padding: '20px'
                }}
                onClick={onClose}
            >
                <motion.div
                    initial={{ scale: 0.9, y: 20 }}
                    animate={{ scale: 1, y: 0 }}
                    exit={{ scale: 0.9, y: 20 }}
                    onClick={(e) => e.stopPropagation()}
                    style={{
                        width: '100%',
                        maxWidth: '900px',
                        maxHeight: '80vh',
                        background: 'var(--color-graphite-gray)',
                        borderRadius: '16px',
                        border: '1px solid rgba(255, 255, 255, 0.1)',
                        display: 'flex',
                        flexDirection: 'column',
                        overflow: 'hidden'
                    }}
                >
                    {/* Header */}
                    <div style={{
                        padding: '24px',
                        borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center'
                    }}>
                        <div>
                            <h2 style={{ fontSize: '1.5rem', fontWeight: 600, marginBottom: '4px' }}>
                                {title}
                            </h2>
                            <p style={{ color: 'var(--color-fog-silver)', fontSize: '0.875rem' }}>
                                Choose a project to compare
                            </p>
                        </div>
                        <button
                            onClick={onClose}
                            style={{
                                padding: '8px',
                                borderRadius: '8px',
                                background: 'transparent',
                                border: 'none',
                                color: 'var(--color-fog-silver)',
                                cursor: 'pointer',
                                transition: 'all 0.2s'
                            }}
                            onMouseEnter={(e) => {
                                e.currentTarget.style.background = 'rgba(255, 255, 255, 0.1)';
                                e.currentTarget.style.color = 'white';
                            }}
                            onMouseLeave={(e) => {
                                e.currentTarget.style.background = 'transparent';
                                e.currentTarget.style.color = 'var(--color-fog-silver)';
                            }}
                        >
                            <X size={24} />
                        </button>
                    </div>

                    {/* Content */}
                    <div style={{ flex: 1, overflowY: 'auto', padding: '24px' }}>
                        {loading && (
                            <div style={{ textAlign: 'center', padding: '40px' }}>
                                <div className="w-8 h-8 border-2 border-soft-gold border-t-transparent rounded-full animate-spin mx-auto" style={{ width: '32px', height: '32px', border: '2px solid var(--color-soft-gold)', borderTopColor: 'transparent', borderRadius: '50%', animation: 'spin 1s linear infinite', margin: '0 auto' }} />
                                <p style={{ marginTop: '16px', color: 'var(--color-fog-silver)' }}>Loading projects...</p>
                            </div>
                        )}

                        {error && (
                            <div style={{
                                padding: '20px',
                                background: 'rgba(255, 0, 0, 0.1)',
                                border: '1px solid rgba(255, 0, 0, 0.3)',
                                borderRadius: '8px',
                                color: '#ff6b6b',
                                textAlign: 'center'
                            }}>
                                {error}
                            </div>
                        )}

                        {!loading && !error && projects.length === 0 && (
                            <div style={{ textAlign: 'center', padding: '40px', color: 'var(--color-fog-silver)' }}>
                                <ImageIcon size={48} style={{ margin: '0 auto 16px' }} />
                                <p>No completed projects found.</p>
                                <p style={{ fontSize: '0.875rem', marginTop: '8px' }}>Create some 3D models first!</p>
                            </div>
                        )}

                        {!loading && !error && projects.length > 0 && (
                            <div style={{
                                display: 'grid',
                                gridTemplateColumns: 'repeat(auto-fill, minmax(250px, 1fr))',
                                gap: '16px'
                            }}>
                                {projects.map((project) => (
                                    <motion.div
                                        key={project.id}
                                        whileHover={{ scale: 1.02 }}
                                        whileTap={{ scale: 0.98 }}
                                        onClick={() => onSelect(project)}
                                        style={{
                                            background: selectedProjectId === project.id 
                                                ? 'rgba(212, 168, 87, 0.2)' 
                                                : 'var(--color-carbon-black)',
                                            border: selectedProjectId === project.id
                                                ? '2px solid var(--color-soft-gold)'
                                                : '1px solid rgba(255, 255, 255, 0.1)',
                                            borderRadius: '12px',
                                            padding: '12px',
                                            cursor: 'pointer',
                                            transition: 'all 0.2s',
                                            position: 'relative'
                                        }}
                                    >
                                        {/* Thumbnail */}
                                        <div style={{
                                            width: '100%',
                                            height: '150px',
                                            background: 'var(--color-graphite-gray)',
                                            borderRadius: '8px',
                                            marginBottom: '12px',
                                            overflow: 'hidden',
                                            display: 'flex',
                                            alignItems: 'center',
                                            justifyContent: 'center'
                                        }}>
                                            {(project.processed_image_url || project.original_image_url) ? (
                                                <img
                                                    src={project.processed_image_url || project.original_image_url || ''}
                                                    alt="Project thumbnail"
                                                    style={{
                                                        width: '100%',
                                                        height: '100%',
                                                        objectFit: 'cover'
                                                    }}
                                                    onError={(e) => {
                                                        e.currentTarget.style.display = 'none';
                                                        if (e.currentTarget.parentElement) {
                                                            const placeholder = document.createElement('div');
                                                            placeholder.innerHTML = '<ImageIcon size={32} color="var(--color-fog-silver)" />';
                                                            e.currentTarget.parentElement.appendChild(placeholder);
                                                        }
                                                    }}
                                                />
                                            ) : (
                                                <ImageIcon size={32} color="var(--color-fog-silver)" />
                                            )}
                                        </div>

                                        {/* Info */}
                                        <div style={{ fontSize: '0.75rem', color: 'var(--color-fog-silver)' }}>
                                            <div style={{ display: 'flex', alignItems: 'center', gap: '4px', marginBottom: '6px' }}>
                                                <Calendar size={12} />
                                                <span>{formatDate(project.created_at)}</span>
                                            </div>
                                            
                                            <div style={{ display: 'flex', alignItems: 'center', gap: '4px', marginBottom: '6px' }}>
                                                <Zap size={12} />
                                                <span>{formatTime(project.total_processing_time)}</span>
                                            </div>

                                            {project.model_info?.num_splats && (
                                                <div style={{ fontSize: '0.7rem', color: 'var(--color-soft-gold)', fontFamily: 'var(--font-mono)' }}>
                                                    {project.model_info.num_splats.toLocaleString()} splats
                                                </div>
                                            )}

                                            {project.quality_metrics?.overall_quality && (
                                                <div style={{ 
                                                    marginTop: '8px',
                                                    padding: '4px 8px',
                                                    background: 'rgba(17, 197, 217, 0.1)',
                                                    borderRadius: '4px',
                                                    fontSize: '0.7rem',
                                                    color: 'var(--color-ultramarine-cyan)',
                                                    textAlign: 'center'
                                                }}>
                                                    Quality: {(project.quality_metrics.overall_quality * 100).toFixed(0)}%
                                                </div>
                                            )}
                                        </div>

                                        {/* Selected Badge */}
                                        {selectedProjectId === project.id && (
                                            <div style={{
                                                position: 'absolute',
                                                top: '8px',
                                                right: '8px',
                                                width: '24px',
                                                height: '24px',
                                                borderRadius: '50%',
                                                background: 'var(--color-soft-gold)',
                                                display: 'flex',
                                                alignItems: 'center',
                                                justifyContent: 'center',
                                                color: 'black',
                                                fontSize: '0.75rem',
                                                fontWeight: 'bold'
                                            }}>
                                                ✓
                                            </div>
                                        )}
                                    </motion.div>
                                ))}
                            </div>
                        )}
                    </div>
                </motion.div>
            </motion.div>
        </AnimatePresence>
    );
};
