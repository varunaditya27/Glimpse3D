import { useState } from 'react';
import { ThreeCanvas } from '../components/ThreeCanvas';
import { Button } from '../components/Button';
import { Upload, Layers, Settings, Share2, Download, Maximize2, RotateCcw, Wand2 } from 'lucide-react';
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
    const [activeTool, setActiveTool] = useState('upload');
    const [status, setStatus] = useState<'idle' | 'uploading' | 'processing' | 'ready' | 'enhancing' | 'enhanced'>('idle');
    const [progress, setProgress] = useState(0);
    const [statusText, setStatusText] = useState('');

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
        }, 3000);
    };

    const handleEnhance = () => {
        setStatus('enhancing');
        setStatusText('Initializing AI pipeline...');

        setTimeout(() => setStatusText('Refining textures (SDXL)...'), 1500);
        setTimeout(() => setStatusText('Back-projecting details...'), 3000);
        setTimeout(() => setStatusText('Optimizing Gaussian Splats...'), 4500);

        setTimeout(() => {
            setStatus('enhanced');
            setStatusText('Model Enhanced');
        }, 6000);
    };

    const handleReset = () => {
        setStatus('idle');
        setProgress(0);
        setStatusText('');
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
                    <ToolButton icon={Upload} label="Upload" active={activeTool === 'upload'} onClick={() => setActiveTool('upload')} />
                    <ToolButton icon={Layers} label="Layers" active={activeTool === 'layers'} onClick={() => setActiveTool('layers')} />
                    <ToolButton icon={Wand2} label="Enhance" active={activeTool === 'enhance'} onClick={() => setActiveTool('enhance')} />
                    <ToolButton icon={Settings} label="Settings" active={activeTool === 'settings'} onClick={() => setActiveTool('settings')} />
                </div>

                <div style={{ marginTop: 'auto', display: 'flex', flexDirection: 'column', gap: '16px', width: '100%', padding: '0 8px' }}>
                    <ToolButton icon={Share2} label="Share" />
                </div>
            </div>

            {/* Center - Canvas */}
            <div className="canvas-area">
                <ThreeCanvas>
                    {/* 3D Content based on state */}
                    {status === 'idle' || status === 'uploading' ? null : (
                        <group>
                            <mesh rotation={[0.5, 0.5, 0]}>
                                <torusKnotGeometry args={[0.45, 0.15, 128, 32]} />
                                <meshStandardMaterial
                                    color={status === 'enhanced' ? "#11C5D9" : "#D4A857"}
                                    roughness={status === 'enhanced' ? 0.1 : 0.5}
                                    metalness={status === 'enhanced' ? 1.0 : 0.5}
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
                {(status === 'uploading' || status === 'processing') && (
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
                            {status === 'processing' && (
                                <div className="w-8 h-8 border-2 border-soft-gold border-t-transparent rounded-full animate-spin mx-auto" style={{ width: '32px', height: '32px', border: '2px solid var(--color-soft-gold)', borderTopColor: 'transparent', borderRadius: '50%', animation: 'spin 1s linear infinite', margin: '0 auto' }} />
                            )}
                        </div>
                    </div>
                )}

                {/* Scanning Effect Overlay */}
                <AnimatePresence>
                    {status === 'enhancing' && (
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

            {/* Right Panel - Details */}
            <div className="details-panel">
                <div className="panel-header">
                    <h3 style={{ color: 'var(--color-pure-white)', fontWeight: 600, marginBottom: '4px' }}>Properties</h3>
                    <p style={{ color: 'var(--color-fog-silver)', fontSize: '0.75rem' }}>Model Metadata & Settings</p>
                </div>

                <div className="panel-content">
                    <div className="prop-group">
                        <label className="prop-label">Dimensions</label>
                        <div className="prop-grid">
                            {['X', 'Y', 'Z'].map(axis => (
                                <div key={axis} className="prop-item">
                                    <span style={{ color: 'var(--color-soft-gold)', fontSize: '0.75rem', marginRight: '4px' }}>{axis}</span>
                                    <span style={{ fontSize: '0.875rem', fontFamily: 'var(--font-mono)', color: 'white' }}>1.00</span>
                                </div>
                            ))}
                        </div>
                    </div>

                    <div className="prop-group">
                        <label className="prop-label">Material</label>
                        <div className="prop-card">
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                <span style={{ fontSize: '0.875rem', color: 'white' }}>Roughness</span>
                                <span style={{ fontSize: '0.75rem', fontFamily: 'var(--font-mono)', color: 'var(--color-soft-gold)' }}>
                                    {status === 'enhanced' ? '0.1' : '0.5'}
                                </span>
                            </div>
                            <div className="bar-container">
                                <motion.div
                                    className="bar-fill"
                                    animate={{ width: status === 'enhanced' ? '10%' : '50%' }}
                                    transition={{ duration: 1 }}
                                />
                            </div>

                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', paddingTop: '8px' }}>
                                <span style={{ fontSize: '0.875rem', color: 'white' }}>Metalness</span>
                                <span style={{ fontSize: '0.75rem', fontFamily: 'var(--font-mono)', color: 'var(--color-soft-gold)' }}>
                                    {status === 'enhanced' ? '1.0' : '0.5'}
                                </span>
                            </div>
                            <div className="bar-container">
                                <motion.div
                                    className="bar-fill"
                                    animate={{ width: status === 'enhanced' ? '100%' : '50%' }}
                                    transition={{ duration: 1 }}
                                />
                            </div>
                        </div>
                    </div>
                </div>

                <div className="panel-footer">
                    <Button variant="outline" disabled={status !== 'enhanced'} style={{ width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}>
                        <Download size={18} />
                        Export Model
                    </Button>
                </div>
            </div>
        </div>
    );
};
