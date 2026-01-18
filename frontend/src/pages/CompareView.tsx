import { useState, useEffect } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { DualThreeCanvas } from '../components/DualThreeCanvas';
import { ProjectSelector } from '../components/ProjectSelector';
import { Button } from '../components/Button';
import { ArrowLeft, GitCompare, BarChart3, Maximize2, RotateCcw, Download } from 'lucide-react';
import { motion } from 'framer-motion';
import clsx from 'clsx';

interface ProjectData {
    id: string;
    status: string;
    original_image_url: string | null;
    model: {
        url: string | null;
        num_splats?: number;
        file_size_mb?: number;
    };
    metrics?: {
        overall_quality?: number;
        psnr?: number;
        ssim?: number;
    } | null;
    total_processing_time?: number;
}

export const CompareView = () => {
    const navigate = useNavigate();
    const [searchParams, setSearchParams] = useSearchParams();
    
    const [projectA, setProjectA] = useState<ProjectData | null>(null);
    const [projectB, setProjectB] = useState<ProjectData | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    
    const [selectorOpen, setSelectorOpen] = useState(false);
    const [selectingFor, setSelectingFor] = useState<'A' | 'B'>('A');
    const [syncCameras, setSyncCameras] = useState(false);
    const [showMetrics, setShowMetrics] = useState(true);

    // Load projects from URL params on mount
    useEffect(() => {
        const idA = searchParams.get('projectA');
        const idB = searchParams.get('projectB');
        
        if (idA && idB) {
            loadComparisonData(idA, idB);
        } else if (!projectA) {
            // Open selector for project A if no params
            openSelector('A');
        }
    }, []);

    const loadComparisonData = async (idA: string, idB: string) => {
        setLoading(true);
        setError(null);
        
        try {
            const response = await fetch(`http://localhost:8000/api/v1/compare/${idA}/${idB}`);
            
            if (!response.ok) {
                throw new Error('Failed to load comparison data');
            }
            
            const data = await response.json();
            setProjectA(data.project_a);
            setProjectB(data.project_b);
            
            // Update URL
            setSearchParams({ projectA: idA, projectB: idB });
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to load projects');
        } finally {
            setLoading(false);
        }
    };

    const openSelector = (target: 'A' | 'B') => {
        setSelectingFor(target);
        setSelectorOpen(true);
    };

    const handleProjectSelect = (project: any) => {
        const projectData: ProjectData = {
            id: project.id,
            status: project.status,
            original_image_url: project.original_image_url,
            model: {
                url: project.latest_model_url,
                num_splats: project.model_info?.num_splats,
                file_size_mb: project.model_info?.file_size_mb
            },
            metrics: project.quality_metrics,
            total_processing_time: project.total_processing_time
        };
        
        if (selectingFor === 'A') {
            setProjectA(projectData);
            
            // If B exists, load full comparison
            if (projectB) {
                loadComparisonData(projectData.id, projectB.id);
            } else {
                // Otherwise, open selector for B
                setTimeout(() => openSelector('B'), 500);
            }
        } else {
            setProjectB(projectData);
            
            // If A exists, load full comparison
            if (projectA) {
                loadComparisonData(projectA.id, projectData.id);
            }
        }
        
        setSelectorOpen(false);
    };

    const formatTime = (seconds?: number) => {
        if (!seconds) return 'N/A';
        if (seconds < 60) return `${Math.round(seconds)}s`;
        return `${Math.round(seconds / 60)}m`;
    };

    const MetricBadge = ({ label, value, color = 'var(--color-soft-gold)' }: { label: string; value: string | number; color?: string }) => (
        <div style={{
            padding: '8px 12px',
            background: 'rgba(11, 12, 16, 0.8)',
            backdropFilter: 'blur(10px)',
            borderRadius: '8px',
            border: `1px solid ${color}30`,
            display: 'flex',
            flexDirection: 'column',
            gap: '2px'
        }}>
            <span style={{ fontSize: '0.7rem', color: 'var(--color-fog-silver)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                {label}
            </span>
            <span style={{ fontSize: '0.9rem', color: color, fontFamily: 'var(--font-mono)', fontWeight: 600 }}>
                {value}
            </span>
        </div>
    );

    return (
        <div className="workspace-container">
            {/* Top Bar */}
            <div style={{
                position: 'absolute',
                top: 0,
                left: 0,
                right: 0,
                height: '64px',
                background: 'rgba(11, 12, 16, 0.95)',
                backdropFilter: 'blur(20px)',
                borderBottom: '1px solid var(--color-graphite-gray)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '0 24px',
                zIndex: 100
            }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                    <button
                        onClick={() => navigate('/workspace')}
                        style={{
                            padding: '8px',
                            borderRadius: '8px',
                            background: 'transparent',
                            border: 'none',
                            color: 'var(--color-fog-silver)',
                            cursor: 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            transition: 'all 0.2s'
                        }}
                        onMouseEnter={(e) => {
                            e.currentTarget.style.background = 'var(--color-graphite-gray)';
                            e.currentTarget.style.color = 'white';
                        }}
                        onMouseLeave={(e) => {
                            e.currentTarget.style.background = 'transparent';
                            e.currentTarget.style.color = 'var(--color-fog-silver)';
                        }}
                    >
                        <ArrowLeft size={20} />
                    </button>
                    
                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                        <GitCompare size={24} color="var(--color-soft-gold)" />
                        <div>
                            <h1 style={{ fontSize: '1.25rem', fontWeight: 600 }}>Model Comparison</h1>
                            <p style={{ fontSize: '0.75rem', color: 'var(--color-fog-silver)' }}>
                                Side-by-side analysis
                            </p>
                        </div>
                    </div>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                    <button
                        onClick={() => setShowMetrics(!showMetrics)}
                        className="icon-btn"
                        title="Toggle Metrics"
                    >
                        <BarChart3 size={20} />
                    </button>
                    
                    <button
                        onClick={() => setSyncCameras(!syncCameras)}
                        className={clsx('icon-btn', syncCameras && 'active')}
                        title="Sync Cameras"
                        style={{
                            background: syncCameras ? 'var(--color-soft-gold)' : undefined,
                            color: syncCameras ? 'black' : undefined
                        }}
                    >
                        <RotateCcw size={20} />
                    </button>
                    
                    <Button variant="outline" size="sm">
                        <Download size={16} style={{ marginRight: '6px' }} />
                        Export Report
                    </Button>
                </div>
            </div>

            {/* Main Content Area */}
            <div style={{ 
                width: '100%', 
                height: '100vh',
                paddingTop: '64px',
                position: 'relative' 
            }}>
                {loading && (
                    <div style={{
                        position: 'absolute',
                        inset: 0,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        background: 'var(--color-carbon-black)',
                        zIndex: 50
                    }}>
                        <div style={{ textAlign: 'center' }}>
                            <div className="w-8 h-8 border-2 border-soft-gold border-t-transparent rounded-full animate-spin mx-auto" style={{ width: '48px', height: '48px', border: '3px solid var(--color-soft-gold)', borderTopColor: 'transparent', borderRadius: '50%', animation: 'spin 1s linear infinite', margin: '0 auto 16px' }} />
                            <p style={{ color: 'var(--color-fog-silver)' }}>Loading comparison...</p>
                        </div>
                    </div>
                )}

                {error && (
                    <div style={{
                        position: 'absolute',
                        inset: 0,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        background: 'var(--color-carbon-black)',
                        zIndex: 50
                    }}>
                        <div style={{
                            padding: '32px',
                            background: 'var(--color-graphite-gray)',
                            borderRadius: '16px',
                            border: '1px solid rgba(255, 0, 0, 0.3)',
                            textAlign: 'center',
                            maxWidth: '400px'
                        }}>
                            <p style={{ color: '#ff6b6b', marginBottom: '16px' }}>{error}</p>
                            <Button onClick={() => navigate('/workspace')}>
                                Back to Workspace
                            </Button>
                        </div>
                    </div>
                )}

                {!loading && !error && (!projectA || !projectB) && (
                    <div style={{
                        position: 'absolute',
                        inset: 0,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        background: 'var(--color-carbon-black)'
                    }}>
                        <div style={{ textAlign: 'center', maxWidth: '500px', padding: '32px' }}>
                            <GitCompare size={64} color="var(--color-soft-gold)" style={{ margin: '0 auto 24px' }} />
                            <h2 style={{ fontSize: '1.5rem', marginBottom: '12px' }}>Select Projects to Compare</h2>
                            <p style={{ color: 'var(--color-fog-silver)', marginBottom: '24px' }}>
                                Choose two 3D models to view them side-by-side
                            </p>
                            
                            <div style={{ display: 'flex', gap: '12px', justifyContent: 'center' }}>
                                <Button onClick={() => openSelector('A')} variant={projectA ? 'outline' : 'primary'}>
                                    {projectA ? 'Change' : 'Select'} Project A
                                </Button>
                                <Button onClick={() => openSelector('B')} variant={projectB ? 'outline' : 'primary'} disabled={!projectA}>
                                    {projectB ? 'Change' : 'Select'} Project B
                                </Button>
                            </div>
                        </div>
                    </div>
                )}

                {!loading && !error && projectA && projectB && (
                    <>
                        <DualThreeCanvas
                            leftModelUrl={projectA.model.url}
                            rightModelUrl={projectB.model.url}
                            syncCameras={syncCameras}
                            leftTitle="Project A"
                            rightTitle="Project B"
                        />

                        {/* Metrics Overlay */}
                        {showMetrics && (
                            <motion.div
                                initial={{ y: 100, opacity: 0 }}
                                animate={{ y: 0, opacity: 1 }}
                                exit={{ y: 100, opacity: 0 }}
                                style={{
                                    position: 'absolute',
                                    bottom: '24px',
                                    left: '50%',
                                    transform: 'translateX(-50%)',
                                    display: 'flex',
                                    gap: '24px',
                                    zIndex: 10
                                }}
                            >
                                {/* Project A Metrics */}
                                <div style={{ display: 'flex', gap: '8px' }}>
                                    {projectA.model.num_splats && (
                                        <MetricBadge 
                                            label="Splats A" 
                                            value={projectA.model.num_splats.toLocaleString()} 
                                            color="var(--color-soft-gold)"
                                        />
                                    )}
                                    {projectA.metrics?.overall_quality && (
                                        <MetricBadge 
                                            label="Quality A" 
                                            value={`${(projectA.metrics.overall_quality * 100).toFixed(0)}%`} 
                                            color="var(--color-soft-gold)"
                                        />
                                    )}
                                    {projectA.total_processing_time && (
                                        <MetricBadge 
                                            label="Time A" 
                                            value={formatTime(projectA.total_processing_time)} 
                                            color="var(--color-soft-gold)"
                                        />
                                    )}
                                </div>

                                {/* Divider */}
                                <div style={{
                                    width: '2px',
                                    background: 'linear-gradient(to bottom, transparent, var(--color-graphite-gray), transparent)'
                                }} />

                                {/* Project B Metrics */}
                                <div style={{ display: 'flex', gap: '8px' }}>
                                    {projectB.model.num_splats && (
                                        <MetricBadge 
                                            label="Splats B" 
                                            value={projectB.model.num_splats.toLocaleString()} 
                                            color="var(--color-ultramarine-cyan)"
                                        />
                                    )}
                                    {projectB.metrics?.overall_quality && (
                                        <MetricBadge 
                                            label="Quality B" 
                                            value={`${(projectB.metrics.overall_quality * 100).toFixed(0)}%`} 
                                            color="var(--color-ultramarine-cyan)"
                                        />
                                    )}
                                    {projectB.total_processing_time && (
                                        <MetricBadge 
                                            label="Time B" 
                                            value={formatTime(projectB.total_processing_time)} 
                                            color="var(--color-ultramarine-cyan)"
                                        />
                                    )}
                                </div>
                            </motion.div>
                        )}

                        {/* Change Projects Buttons */}
                        <div style={{
                            position: 'absolute',
                            top: '80px',
                            left: '24px',
                            zIndex: 10
                        }}>
                            <Button variant="outline" size="sm" onClick={() => openSelector('A')}>
                                Change Project A
                            </Button>
                        </div>
                        
                        <div style={{
                            position: 'absolute',
                            top: '80px',
                            right: '24px',
                            zIndex: 10
                        }}>
                            <Button variant="outline" size="sm" onClick={() => openSelector('B')}>
                                Change Project B
                            </Button>
                        </div>
                    </>
                )}
            </div>

            {/* Project Selector Modal */}
            <ProjectSelector
                isOpen={selectorOpen}
                onClose={() => setSelectorOpen(false)}
                onSelect={handleProjectSelect}
                selectedProjectId={selectingFor === 'A' ? projectA?.id : projectB?.id}
                title={`Select Project ${selectingFor}`}
            />
        </div>
    );
};
