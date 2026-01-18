import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ThreeCanvas } from '../components/ThreeCanvas';
import { Button } from '../components/Button';
import { Upload, Sliders, Sparkles, Download, Maximize2, RotateCcw, Wand2, Folder, GitCompare } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import clsx from 'clsx';

const ToolButton = ({ icon: Icon, label, active, onClick }: { icon: any, label: string, active?: boolean, onClick?: () => void }) => (
    <button
        onClick={onClick}
        className={clsx('sidebar-btn', active && 'active')}
        title={label}
    >
        <Icon size={24} strokeWidth={1.5} />
        {active && (
            <motion.div
                layoutId="active-indicator"
                className="active-indicator"
                transition={{ type: "spring", stiffness: 300, damping: 30 }}
            />
        )}
    </button>
);

export const Workspace = () => {
    const navigate = useNavigate();
    
    // Tools: 'project', 'editor', 'ai', 'export'
    const [activeTool, setActiveTool] = useState('project');
    const [status, setStatus] = useState<'idle' | 'uploading' | 'processing' | 'ready' | 'enhancing' | 'enhanced'>('idle');
    const [progress, setProgress] = useState(0);
    const [statusText, setStatusText] = useState('');

    // Material State
    const [material, setMaterial] = useState({
        color: '#D4A857',
        roughness: 0.2,
        metalness: 0.9,
        emissive: 0,
        wireframe: false,
        autoRotate: false
    });

    // AI Command Bar State
    const [prompt, setPrompt] = useState('');
    const [isGenerating, setIsGenerating] = useState(false);

    const handleUpload = () => {
        setStatus('uploading');
        setProgress(0);
        setStatusText('Uploading image...');

        // Simulate Upload
        const interval = setInterval(() => {
            setProgress(prev => {
                if (prev >= 100) {
                    clearInterval(interval);
                    startProcessing();
                    return 100;
                }
                return prev + 5;
            });
        }, 100);
    };

    const startProcessing = () => {
        setStatus('processing');
        setStatusText('Generating coarse 3D model...');
        setTimeout(() => {
            setStatus('ready');
            setStatusText('Ready to enhance');
            setActiveTool('editor'); // Auto-switch to editor
        }, 3000);
    };

    const handleEnhance = () => {
        setStatus('enhancing');
        setStatusText('Initializing AI pipeline...');
        setActiveTool('ai'); // Auto-switch to AI view

        setTimeout(() => setStatusText('Refining textures (SDXL)...'), 1500);
        setTimeout(() => setStatusText('Back-projecting details...'), 3000);
        setTimeout(() => setStatusText('Optimizing Gaussian Splats...'), 4500);

        setTimeout(() => {
            setStatus('enhanced');
            setStatusText('Model Enhanced');
            // Update material to "premium" look
            setMaterial(prev => ({ ...prev, color: '#11C5D9', roughness: 0.1, metalness: 1.0, emissive: 0.2 }));
        }, 6000);
    };

    const handleReset = () => {
        setStatus('idle');
        setProgress(0);
        setStatusText('');
        setMaterial({
            color: '#D4A857',
            roughness: 0.2,
            metalness: 0.9,
            emissive: 0,
            wireframe: false,
            autoRotate: false
        });
        setPrompt('');
        setActiveTool('project');
    };

    const handlePromptSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (!prompt.trim()) return;

        setIsGenerating(true);
        setStatusText('AI is editing model...');

        // Mock AI Logic
        setTimeout(() => {
            const p = prompt.toLowerCase();
            let newMaterial = { ...material };

            if (p.includes('gold')) {
                newMaterial = { ...material, color: '#FFD700', roughness: 0.2, metalness: 1.0 };
            } else if (p.includes('rusty') || p.includes('rust')) {
                newMaterial = { ...material, color: '#8B4513', roughness: 0.8, metalness: 0.2 };
            } else if (p.includes('blue') || p.includes('ice')) {
                newMaterial = { ...material, color: '#00FFFF', roughness: 0.1, metalness: 0.8, emissive: 0.5 };
            } else if (p.includes('stone') || p.includes('rock')) {
                newMaterial = { ...material, color: '#808080', roughness: 0.9, metalness: 0.1 };
            } else {
                // Default "Cyberpunk" random
                newMaterial = { ...material, color: '#FF00FF', roughness: 0.2, metalness: 0.8 };
            }

            setMaterial(newMaterial);
            setIsGenerating(false);
            setStatusText('Model updated via AI');
            setPrompt('');
        }, 2000);
    };

    return (
        <div className="workspace-container">

            {/* Left Rail - Tools */}
            <div className="sidebar">
                <div style={{ marginBottom: '32px' }}>
                    <div style={{ width: '40px', height: '40px', borderRadius: '8px', background: 'linear-gradient(135deg, var(--color-soft-gold), var(--color-graphite-gray))', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 'bold', color: 'black', boxShadow: '0 0 15px rgba(212, 168, 87, 0.3)' }}>
                        G3D
                    </div>
                </div>

                <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', width: '100%', padding: '0 8px' }}>
                    <ToolButton icon={Folder} label="Project" active={activeTool === 'project'} onClick={() => setActiveTool('project')} />
                    <ToolButton icon={Sliders} label="Editor" active={activeTool === 'editor'} onClick={() => setActiveTool('editor')} />
                    <ToolButton icon={Sparkles} label="AI Lab" active={activeTool === 'ai'} onClick={() => setActiveTool('ai')} />
                    <ToolButton icon={Download} label="Export" active={activeTool === 'export'} onClick={() => setActiveTool('export')} />
                </div>
            </div>

            {/* Center - Canvas */}
            <div className="canvas-area">
                <ThreeCanvas autoRotate={material.autoRotate}>
                    {/* 3D Content based on state */}
                    {status === 'idle' || status === 'uploading' ? null : (
                        <group>
                            <mesh rotation={[0.5, 0.5, 0]}>
                                <torusKnotGeometry args={[0.45, 0.15, 128, 32]} />
                                <meshStandardMaterial
                                    color={material.color}
                                    roughness={material.roughness}
                                    metalness={material.metalness}
                                    emissive={material.color}
                                    emissiveIntensity={material.emissive}
                                    wireframe={material.wireframe}
                                />
                            </mesh>
                        </group>
                    )}
                </ThreeCanvas>

                {/* Empty State / Upload Prompt */}
                {status === 'idle' && (
                    <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column', gap: '24px' }}>
                        <div style={{ width: '400px', padding: '48px', border: '2px dashed var(--color-graphite-gray)', borderRadius: '16px', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '16px', backgroundColor: 'rgba(11, 12, 16, 0.8)' }}>
                            <Upload size={48} color="var(--color-fog-silver)" />
                            <div style={{ textAlign: 'center' }}>
                                <h3 style={{ fontSize: '1.25rem', marginBottom: '8px' }}>Upload an Image</h3>
                                <p style={{ color: 'var(--color-fog-silver)', fontSize: '0.875rem' }}>JPG or PNG. We'll turn it into 3D.</p>
                            </div>
                            <Button onClick={handleUpload}>Select File</Button>
                        </div>
                    </div>
                )}

                {/* Uploading / Processing Overlay */}
                {(status === 'uploading' || status === 'processing' || isGenerating) && (
                    <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', backgroundColor: 'rgba(0,0,0,0.7)', backdropFilter: 'blur(5px)', zIndex: 20 }}>
                        <div style={{ textAlign: 'center', width: '300px' }}>
                            <h3 style={{ marginBottom: '16px', fontSize: '1.25rem' }}>{statusText}</h3>
                            {status === 'uploading' && (
                                <div style={{ width: '100%', height: '4px', background: 'var(--color-graphite-gray)', borderRadius: '2px', overflow: 'hidden' }}>
                                    <motion.div
                                        style={{ width: '100%', height: '100%', background: 'var(--color-soft-gold)', transformOrigin: 'left' }}
                                        initial={{ scaleX: 0 }}
                                        animate={{ scaleX: progress / 100 }}
                                    />
                                </div>
                            )}
                            {(status === 'processing' || isGenerating) && (
                                <div className="w-8 h-8 border-2 border-soft-gold border-t-transparent rounded-full animate-spin mx-auto" style={{ width: '32px', height: '32px', border: '2px solid var(--color-soft-gold)', borderTopColor: 'transparent', borderRadius: '50%', animation: 'spin 1s linear infinite', margin: '0 auto' }} />
                            )}
                        </div>
                    </div>
                )}

                {/* Scanning Effect Overlay */}
                <AnimatePresence>
                    {(status === 'enhancing' || isGenerating) && (
                        <motion.div
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                            style={{ position: 'absolute', inset: 0, pointerEvents: 'none', zIndex: 5, background: 'linear-gradient(to bottom, transparent 49%, rgba(17, 197, 217, 0.5) 50%, transparent 51%)', backgroundSize: '100% 200%', animation: 'scan 2s linear infinite' }}
                        />
                    )}
                </AnimatePresence>

                {/* Canvas Overlays (Visible when model is present) */}
                {(status === 'ready' || status === 'enhancing' || status === 'enhanced') && (
                    <>
                        <div className="canvas-overlay-tl">
                            <h2 style={{ color: 'var(--color-pure-white)', fontSize: '1.5rem', fontWeight: 600, letterSpacing: '-0.02em' }}>Demo Model</h2>
                            <p style={{ color: 'var(--color-fog-silver)', fontSize: '0.875rem', display: 'flex', alignItems: 'center', gap: '6px' }}>
                                <span style={{ width: '6px', height: '6px', borderRadius: '50%', backgroundColor: status === 'enhanced' ? 'var(--color-soft-gold)' : 'var(--color-ultramarine-cyan)', boxShadow: `0 0 8px ${status === 'enhanced' ? 'var(--color-soft-gold)' : 'var(--color-ultramarine-cyan)'}` }}></span>
                                {statusText}
                            </p>
                        </div>

                        {/* AI Command Bar - Visible in Editor and AI modes */}
                        {(activeTool === 'editor' || activeTool === 'ai') && (
                            <div style={{ position: 'absolute', bottom: '100px', left: '50%', transform: 'translateX(-50%)', zIndex: 15, width: '400px' }}>
                                <form onSubmit={handlePromptSubmit} style={{ position: 'relative', width: '100%' }}>
                                    <input
                                        type="text"
                                        placeholder="Describe changes (e.g. 'Make it gold')..."
                                        value={prompt}
                                        onChange={(e) => setPrompt(e.target.value)}
                                        style={{
                                            width: '100%',
                                            padding: '12px 48px 12px 20px',
                                            borderRadius: '30px',
                                            border: '1px solid rgba(255,255,255,0.1)',
                                            background: 'rgba(11, 12, 16, 0.8)',
                                            backdropFilter: 'blur(10px)',
                                            color: 'white',
                                            outline: 'none',
                                            boxShadow: '0 10px 30px -10px rgba(0,0,0,0.5)'
                                        }}
                                    />
                                    <button
                                        type="submit"
                                        style={{
                                            position: 'absolute',
                                            right: '8px',
                                            top: '50%',
                                            transform: 'translateY(-50%)',
                                            width: '32px',
                                            height: '32px',
                                            borderRadius: '50%',
                                            background: 'var(--color-soft-gold)',
                                            display: 'flex',
                                            alignItems: 'center',
                                            justifyContent: 'center',
                                            color: 'black',
                                            cursor: 'pointer',
                                            border: 'none'
                                        }}
                                    >
                                        <Wand2 size={16} />
                                    </button>
                                </form>
                            </div>
                        )}

                        <div className="canvas-overlay-bc">
                            {status === 'ready' && (
                                <Button
                                    size="lg"
                                    onClick={handleEnhance}
                                    style={{ boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.5)' }}
                                >
                                    <Wand2 size={18} style={{ marginRight: '8px' }} />
                                    Enhance Model
                                </Button>
                            )}
                            {status === 'enhancing' && (
                                <div style={{ padding: '12px 24px', background: 'rgba(11, 12, 16, 0.8)', backdropFilter: 'blur(10px)', borderRadius: '30px', border: '1px solid var(--color-ultramarine-cyan)', color: 'var(--color-ultramarine-cyan)', display: 'flex', alignItems: 'center', gap: '10px' }}>
                                    <div className="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin" style={{ width: '16px', height: '16px', border: '2px solid currentColor', borderTopColor: 'transparent', borderRadius: '50%', animation: 'spin 1s linear infinite' }} />
                                    {statusText}
                                </div>
                            )}
                            {status === 'enhanced' && (
                                <Button
                                    size="lg"
                                    variant="outline"
                                    onClick={handleReset}
                                >
                                    <RotateCcw size={18} style={{ marginRight: '8px' }} />
                                    Start Over
                                </Button>
                            )}
                        </div>

                        <div className="canvas-overlay-tr">
                            <button className="icon-btn" onClick={handleReset} title="Reset Demo">
                                <RotateCcw size={20} />
                            </button>
                            <button className="icon-btn">
                                <Maximize2 size={20} />
                            </button>
                        </div>
                    </>
                )}
            </div>

            {/* Right Panel - Contextual */}
            <div className="details-panel">
                <div className="panel-header">
                    <h3 style={{ color: 'var(--color-pure-white)', fontWeight: 600, marginBottom: '4px' }}>
                        {activeTool === 'project' && 'Project Info'}
                        {activeTool === 'editor' && 'Material Editor'}
                        {activeTool === 'ai' && 'AI Lab'}
                        {activeTool === 'export' && 'Export'}
                    </h3>
                    <p style={{ color: 'var(--color-fog-silver)', fontSize: '0.75rem' }}>
                        {activeTool === 'project' && 'Manage your 3D asset'}
                        {activeTool === 'editor' && 'Fine-tune appearance'}
                        {activeTool === 'ai' && 'Enhance with Generative AI'}
                        {activeTool === 'export' && 'Download your model'}
                    </p>
                </div>

                <div className="panel-content">

                    {/* PROJECT VIEW */}
                    {activeTool === 'project' && (
                        <div className="prop-group">
                            <label className="prop-label">Model Metadata</label>
                            <div className="prop-card">
                                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                                    <span style={{ fontSize: '0.875rem', color: 'var(--color-fog-silver)' }}>Status</span>
                                    <span style={{ fontSize: '0.875rem', fontFamily: 'var(--font-mono)', color: status === 'enhanced' ? 'var(--color-soft-gold)' : 'white' }}>{status.toUpperCase()}</span>
                                </div>
                                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                                    <span style={{ fontSize: '0.875rem', color: 'var(--color-fog-silver)' }}>Vertices</span>
                                    <span style={{ fontSize: '0.875rem', fontFamily: 'var(--font-mono)' }}>12,405</span>
                                </div>
                                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                    <span style={{ fontSize: '0.875rem', color: 'var(--color-fog-silver)' }}>Format</span>
                                    <span style={{ fontSize: '0.875rem', fontFamily: 'var(--font-mono)' }}>GLB</span>
                                </div>
                            </div>

                            <div style={{ marginTop: '24px' }}>
                                <Button variant="outline" onClick={handleReset} style={{ width: '100%' }}>
                                    <Upload size={16} style={{ marginRight: '8px' }} />
                                    Upload New Image
                                </Button>
                            </div>
                        </div>
                    )}

                    {/* EDITOR VIEW */}
                    {activeTool === 'editor' && (
                        <>
                            <div className="prop-group">
                                <label className="prop-label">Material</label>
                                <div className="prop-card">
                                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                        <span style={{ fontSize: '0.875rem', color: 'white' }}>Base Color</span>
                                        <input
                                            type="color"
                                            value={material.color}
                                            onChange={(e) => setMaterial({ ...material, color: e.target.value })}
                                            style={{ background: 'none', border: 'none', width: '24px', height: '24px', cursor: 'pointer' }}
                                        />
                                    </div>
                                    <div style={{ marginTop: '12px' }}>
                                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                                            <span style={{ fontSize: '0.875rem', color: 'white' }}>Roughness</span>
                                            <span style={{ fontSize: '0.75rem', fontFamily: 'var(--font-mono)', color: 'var(--color-soft-gold)' }}>{material.roughness.toFixed(2)}</span>
                                        </div>
                                        <input
                                            type="range"
                                            min="0" max="1" step="0.01"
                                            value={material.roughness}
                                            onChange={(e) => setMaterial({ ...material, roughness: parseFloat(e.target.value) })}
                                        />
                                    </div>
                                    <div style={{ marginTop: '12px' }}>
                                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                                            <span style={{ fontSize: '0.875rem', color: 'white' }}>Metalness</span>
                                            <span style={{ fontSize: '0.75rem', fontFamily: 'var(--font-mono)', color: 'var(--color-soft-gold)' }}>{material.metalness.toFixed(2)}</span>
                                        </div>
                                        <input
                                            type="range"
                                            min="0" max="1" step="0.01"
                                            value={material.metalness}
                                            onChange={(e) => setMaterial({ ...material, metalness: parseFloat(e.target.value) })}
                                        />
                                    </div>
                                </div>
                            </div>

                            <div className="prop-group">
                                <label className="prop-label">View Settings</label>
                                <div className="prop-card">
                                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                        <span style={{ fontSize: '0.875rem', color: 'white' }}>Auto Rotate</span>
                                        <label className="toggle-switch">
                                            <input
                                                type="checkbox"
                                                checked={material.autoRotate}
                                                onChange={(e) => setMaterial({ ...material, autoRotate: e.target.checked })}
                                            />
                                            <span className="slider-toggle"></span>
                                        </label>
                                    </div>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '12px' }}>
                                        <span style={{ fontSize: '0.875rem', color: 'white' }}>Wireframe</span>
                                        <label className="toggle-switch">
                                            <input
                                                type="checkbox"
                                                checked={material.wireframe}
                                                onChange={(e) => setMaterial({ ...material, wireframe: e.target.checked })}
                                            />
                                            <span className="slider-toggle"></span>
                                        </label>
                                    </div>
                                </div>
                            </div>
                        </>
                    )}

                    {/* AI LAB VIEW */}
                    {activeTool === 'ai' && (
                        <div className="prop-group">
                            <label className="prop-label">Enhancement Controls</label>
                            <div className="prop-card">
                                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                                    <span style={{ fontSize: '0.875rem', color: 'white' }}>Denoising Strength</span>
                                    <span style={{ fontSize: '0.75rem', fontFamily: 'var(--font-mono)', color: 'var(--color-soft-gold)' }}>0.75</span>
                                </div>
                                <input type="range" min="0" max="1" step="0.01" defaultValue="0.75" />

                                <div style={{ marginTop: '12px', display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                                    <span style={{ fontSize: '0.875rem', color: 'white' }}>Texture Detail</span>
                                    <span style={{ fontSize: '0.75rem', fontFamily: 'var(--font-mono)', color: 'var(--color-soft-gold)' }}>High</span>
                                </div>
                                <input type="range" min="0" max="1" step="0.5" defaultValue="1" />

                                <div style={{ marginTop: '24px', padding: '12px', background: 'rgba(17, 197, 217, 0.1)', borderRadius: '8px', border: '1px solid rgba(17, 197, 217, 0.3)' }}>
                                    <p style={{ fontSize: '0.75rem', color: 'var(--color-ultramarine-cyan)' }}>
                                        <Sparkles size={12} style={{ display: 'inline', marginRight: '4px' }} />
                                        AI Enhancement is active. Use the command bar to refine further.
                                    </p>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* EXPORT VIEW */}
                    {activeTool === 'export' && (
                        <>
                            <div className="prop-group">
                                <label className="prop-label">Export Options</label>
                                <div className="prop-card">
                                    <div style={{ marginBottom: '12px' }}>
                                        <label style={{ display: 'block', fontSize: '0.875rem', color: 'white', marginBottom: '8px' }}>Format</label>
                                        <select style={{ width: '100%', padding: '8px', borderRadius: '8px', background: 'rgba(0,0,0,0.3)', border: '1px solid var(--color-graphite-gray)', color: 'white' }}>
                                            <option>GLB (Recommended)</option>
                                            <option>OBJ</option>
                                            <option>USDZ</option>
                                            <option>Splat</option>
                                        </select>
                                    </div>
                                    <Button variant="outline" disabled={status !== 'enhanced'} style={{ width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}>
                                        <Download size={18} />
                                        Download Model
                                    </Button>
                                </div>
                            </div>

                            {/* COMPARE MODELS SECTION */}
                            <div className="prop-group">
                                <label className="prop-label">Comparison</label>
                                <div className="prop-card">
                                    <p style={{ fontSize: '0.875rem', color: 'var(--color-fog-silver)', marginBottom: '12px' }}>
                                        View multiple models side-by-side
                                    </p>
                                    <Button 
                                        variant="outline" 
                                        onClick={() => navigate('/compare')}
                                        style={{ width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}
                                    >
                                        <GitCompare size={18} />
                                        Compare Models
                                    </Button>
                                </div>
                            </div>
                        </>
                    )}
                </div>
            </div>
        </div>
    );
};
