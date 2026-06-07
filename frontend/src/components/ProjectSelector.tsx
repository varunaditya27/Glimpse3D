import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Box, Calendar } from 'lucide-react';

// Matches the compare API contract:
// GET /compare/projects -> { projects: [ { id, model_url, created_at, status } ] }
export interface CompareProject {
    id: string;
    model_url: string;
    created_at: string;
    status: string;
}

interface ProjectSelectorProps {
    isOpen: boolean;
    onClose: () => void;
    onSelect: (project: CompareProject) => void;
    selectedProjectId?: string | null;
    title?: string;
    apiUrl: string;
}

export const ProjectSelector = ({
    isOpen,
    onClose,
    onSelect,
    selectedProjectId,
    title = 'Select a Project',
    apiUrl,
}: ProjectSelectorProps) => {
    const [projects, setProjects] = useState<CompareProject[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        if (isOpen) {
            fetchProjects();
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [isOpen]);

    const fetchProjects = async () => {
        setLoading(true);
        setError(null);

        try {
            const response = await fetch(`${apiUrl}/compare/projects?limit=50`);
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
        if (isNaN(date.getTime())) return dateString;
        return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
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
                    padding: '20px',
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
                        overflow: 'hidden',
                    }}
                >
                    {/* Header */}
                    <div
                        style={{
                            padding: '24px',
                            borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
                            display: 'flex',
                            justifyContent: 'space-between',
                            alignItems: 'center',
                        }}
                    >
                        <div>
                            <h2 style={{ fontSize: '1.5rem', fontWeight: 600, marginBottom: '4px' }}>{title}</h2>
                            <p style={{ color: 'var(--color-fog-silver)', fontSize: '0.875rem' }}>
                                Choose a completed model to compare
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
                                transition: 'all 0.2s',
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
                                <div
                                    style={{
                                        width: '32px',
                                        height: '32px',
                                        border: '2px solid var(--color-soft-gold)',
                                        borderTopColor: 'transparent',
                                        borderRadius: '50%',
                                        animation: 'spin 1s linear infinite',
                                        margin: '0 auto',
                                    }}
                                />
                                <p style={{ marginTop: '16px', color: 'var(--color-fog-silver)' }}>
                                    Loading projects...
                                </p>
                            </div>
                        )}

                        {error && (
                            <div
                                style={{
                                    padding: '20px',
                                    background: 'rgba(255, 0, 0, 0.1)',
                                    border: '1px solid rgba(255, 0, 0, 0.3)',
                                    borderRadius: '8px',
                                    color: '#ff6b6b',
                                    textAlign: 'center',
                                }}
                            >
                                {error}
                            </div>
                        )}

                        {!loading && !error && projects.length === 0 && (
                            <div style={{ textAlign: 'center', padding: '40px', color: 'var(--color-fog-silver)' }}>
                                <Box size={48} style={{ margin: '0 auto 16px' }} />
                                <p>No completed projects found.</p>
                                <p style={{ fontSize: '0.875rem', marginTop: '8px' }}>Create some 3D models first!</p>
                            </div>
                        )}

                        {!loading && !error && projects.length > 0 && (
                            <div
                                style={{
                                    display: 'grid',
                                    gridTemplateColumns: 'repeat(auto-fill, minmax(250px, 1fr))',
                                    gap: '16px',
                                }}
                            >
                                {projects.map((project) => (
                                    <motion.div
                                        key={project.id}
                                        whileHover={{ scale: 1.02 }}
                                        whileTap={{ scale: 0.98 }}
                                        onClick={() => onSelect(project)}
                                        style={{
                                            background:
                                                selectedProjectId === project.id
                                                    ? 'rgba(212, 168, 87, 0.2)'
                                                    : 'var(--color-carbon-black)',
                                            border:
                                                selectedProjectId === project.id
                                                    ? '2px solid var(--color-soft-gold)'
                                                    : '1px solid rgba(255, 255, 255, 0.1)',
                                            borderRadius: '12px',
                                            padding: '16px',
                                            cursor: 'pointer',
                                            transition: 'all 0.2s',
                                            position: 'relative',
                                        }}
                                    >
                                        {/* Icon header */}
                                        <div
                                            style={{
                                                width: '100%',
                                                height: '100px',
                                                background: 'var(--color-graphite-gray)',
                                                borderRadius: '8px',
                                                marginBottom: '12px',
                                                display: 'flex',
                                                alignItems: 'center',
                                                justifyContent: 'center',
                                            }}
                                        >
                                            <Box size={40} color="var(--color-soft-gold)" />
                                        </div>

                                        {/* Info */}
                                        <div style={{ fontSize: '0.75rem', color: 'var(--color-fog-silver)' }}>
                                            <div
                                                style={{
                                                    fontFamily: 'var(--font-mono)',
                                                    color: 'white',
                                                    marginBottom: '8px',
                                                    overflow: 'hidden',
                                                    textOverflow: 'ellipsis',
                                                    whiteSpace: 'nowrap',
                                                }}
                                                title={project.id}
                                            >
                                                {project.id}
                                            </div>

                                            <div
                                                style={{
                                                    display: 'flex',
                                                    alignItems: 'center',
                                                    gap: '4px',
                                                    marginBottom: '6px',
                                                }}
                                            >
                                                <Calendar size={12} />
                                                <span>{formatDate(project.created_at)}</span>
                                            </div>

                                            <div
                                                style={{
                                                    marginTop: '8px',
                                                    padding: '4px 8px',
                                                    background: 'rgba(17, 197, 217, 0.1)',
                                                    borderRadius: '4px',
                                                    fontSize: '0.7rem',
                                                    color: 'var(--color-ultramarine-cyan)',
                                                    textAlign: 'center',
                                                    textTransform: 'uppercase',
                                                    letterSpacing: '0.05em',
                                                }}
                                            >
                                                {project.status}
                                            </div>
                                        </div>

                                        {/* Selected Badge */}
                                        {selectedProjectId === project.id && (
                                            <div
                                                style={{
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
                                                    fontWeight: 'bold',
                                                }}
                                            >
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
