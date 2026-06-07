import { useState, useEffect } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { DualThreeCanvas } from '../components/DualThreeCanvas';
import { ProjectSelector } from '../components/ProjectSelector';
import type { CompareProject } from '../components/ProjectSelector';
import { Button } from '../components/Button';
import { ArrowLeft, GitCompare } from 'lucide-react';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Resolve a backend-relative model_url (e.g. "/outputs/<id>/...") against API_URL
// so splats load correctly through a remote tunnel too.
const toAbs = (u?: string | null) => (u && u.startsWith('/') ? `${API_URL}${u}` : u || '');

// Shape returned by GET /compare/{id1}/{id2}: { a: {...}, b: {...} }.
interface CompareResponse {
    a: CompareProject;
    b: CompareProject;
}

export const CompareView = () => {
    const navigate = useNavigate();
    const [searchParams, setSearchParams] = useSearchParams();

    const [projectA, setProjectA] = useState<CompareProject | null>(null);
    const [projectB, setProjectB] = useState<CompareProject | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const [selectorOpen, setSelectorOpen] = useState(false);
    const [selectingFor, setSelectingFor] = useState<'A' | 'B'>('A');

    // Load projects from URL params on mount (if both ids are present).
    useEffect(() => {
        const idA = searchParams.get('projectA');
        const idB = searchParams.get('projectB');

        if (idA && idB) {
            loadComparisonData(idA, idB);
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    const loadComparisonData = async (idA: string, idB: string) => {
        setLoading(true);
        setError(null);

        try {
            const response = await fetch(`${API_URL}/compare/${idA}/${idB}`);

            if (!response.ok) {
                throw new Error('Failed to load comparison data (one or both models not completed)');
            }

            const data: CompareResponse = await response.json();
            setProjectA(data.a);
            setProjectB(data.b);

            // Reflect the selection in the URL.
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

    const handleProjectSelect = (project: CompareProject) => {
        if (selectingFor === 'A') {
            setProjectA(project);
            if (projectB) {
                loadComparisonData(project.id, projectB.id);
            } else {
                setTimeout(() => openSelector('B'), 400);
            }
        } else {
            setProjectB(project);
            if (projectA) {
                loadComparisonData(projectA.id, project.id);
            }
        }

        setSelectorOpen(false);
    };

    return (
        <div className="workspace-container">
            {/* Top Bar */}
            <div
                style={{
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
                    zIndex: 100,
                }}
            >
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
                            transition: 'all 0.2s',
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
                    <Button variant="outline" size="sm" onClick={() => openSelector('A')}>
                        {projectA ? 'Change' : 'Select'} A
                    </Button>
                    <Button
                        variant="outline"
                        size="sm"
                        onClick={() => openSelector('B')}
                        disabled={!projectA}
                    >
                        {projectB ? 'Change' : 'Select'} B
                    </Button>
                </div>
            </div>

            {/* Main Content Area */}
            <div
                style={{
                    width: '100%',
                    height: '100vh',
                    paddingTop: '64px',
                    position: 'relative',
                }}
            >
                {loading && (
                    <div
                        style={{
                            position: 'absolute',
                            inset: 0,
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            background: 'var(--color-carbon-black)',
                            zIndex: 50,
                        }}
                    >
                        <div style={{ textAlign: 'center' }}>
                            <div
                                style={{
                                    width: '48px',
                                    height: '48px',
                                    border: '3px solid var(--color-soft-gold)',
                                    borderTopColor: 'transparent',
                                    borderRadius: '50%',
                                    animation: 'spin 1s linear infinite',
                                    margin: '0 auto 16px',
                                }}
                            />
                            <p style={{ color: 'var(--color-fog-silver)' }}>Loading comparison...</p>
                        </div>
                    </div>
                )}

                {error && (
                    <div
                        style={{
                            position: 'absolute',
                            inset: 0,
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            background: 'var(--color-carbon-black)',
                            zIndex: 50,
                        }}
                    >
                        <div
                            style={{
                                padding: '32px',
                                background: 'var(--color-graphite-gray)',
                                borderRadius: '16px',
                                border: '1px solid rgba(255, 0, 0, 0.3)',
                                textAlign: 'center',
                                maxWidth: '400px',
                            }}
                        >
                            <p style={{ color: '#ff6b6b', marginBottom: '16px' }}>{error}</p>
                            <Button onClick={() => navigate('/workspace')}>Back to Workspace</Button>
                        </div>
                    </div>
                )}

                {!loading && !error && (!projectA || !projectB) && (
                    <div
                        style={{
                            position: 'absolute',
                            inset: 0,
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            background: 'var(--color-carbon-black)',
                        }}
                    >
                        <div style={{ textAlign: 'center', maxWidth: '500px', padding: '32px' }}>
                            <GitCompare
                                size={64}
                                color="var(--color-soft-gold)"
                                style={{ margin: '0 auto 24px' }}
                            />
                            <h2 style={{ fontSize: '1.5rem', marginBottom: '12px' }}>
                                Select Projects to Compare
                            </h2>
                            <p style={{ color: 'var(--color-fog-silver)', marginBottom: '24px' }}>
                                Choose two completed 3D models to view them side-by-side
                            </p>

                            <div style={{ display: 'flex', gap: '12px', justifyContent: 'center' }}>
                                <Button
                                    onClick={() => openSelector('A')}
                                    variant={projectA ? 'outline' : 'primary'}
                                >
                                    {projectA ? 'Change' : 'Select'} Project A
                                </Button>
                                <Button
                                    onClick={() => openSelector('B')}
                                    variant={projectB ? 'outline' : 'primary'}
                                    disabled={!projectA}
                                >
                                    {projectB ? 'Change' : 'Select'} Project B
                                </Button>
                            </div>
                        </div>
                    </div>
                )}

                {!loading && !error && projectA && projectB && (
                    <DualThreeCanvas urlA={toAbs(projectA.model_url)} urlB={toAbs(projectB.model_url)} />
                )}
            </div>

            {/* Project Selector Modal */}
            <ProjectSelector
                isOpen={selectorOpen}
                onClose={() => setSelectorOpen(false)}
                onSelect={handleProjectSelect}
                selectedProjectId={selectingFor === 'A' ? projectA?.id : projectB?.id}
                title={`Select Project ${selectingFor}`}
                apiUrl={API_URL}
            />
        </div>
    );
};
