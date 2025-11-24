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
    const [isEnhancing, setIsEnhancing] = useState(false);

    const handleEnhance = () => {
        setIsEnhancing(true);
        setTimeout(() => setIsEnhancing(false), 3000);
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
                    {/* Placeholder 3D Content */}
                    <mesh rotation={[0.5, 0.5, 0]}>
                        <boxGeometry args={[1, 1, 1]} />
                        <meshStandardMaterial color="#D4A857" roughness={0.2} metalness={0.9} />
                    </mesh>
                </ThreeCanvas>

                {/* Scanning Effect Overlay */}
                <AnimatePresence>
                    {isEnhancing && (
                        <motion.div
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                            style={{ position: 'absolute', inset: 0, pointerEvents: 'none', zIndex: 5, background: 'linear-gradient(to bottom, transparent 49%, rgba(17, 197, 217, 0.5) 50%, transparent 51%)', backgroundSize: '100% 200%', animation: 'scan 2s linear infinite' }}
                        />
                    )}
                </AnimatePresence>

                {/* Canvas Overlays */}
                <div className="canvas-overlay-tl">
                    <h2 style={{ color: 'var(--color-pure-white)', fontSize: '1.5rem', fontWeight: 600, letterSpacing: '-0.02em' }}>Untitled Model</h2>
                    <p style={{ color: 'var(--color-fog-silver)', fontSize: '0.875rem', display: 'flex', alignItems: 'center', gap: '6px' }}>
                        <span style={{ width: '6px', height: '6px', borderRadius: '50%', backgroundColor: 'var(--color-ultramarine-cyan)', boxShadow: '0 0 8px var(--color-ultramarine-cyan)' }}></span>
                        Ready to enhance
                    </p>
                </div>

                <div className="canvas-overlay-bc">
                    <Button
                        size="lg"
                        onClick={handleEnhance}
                        isLoading={isEnhancing}
                        style={{ boxShadow: isEnhancing ? '0 0 40px rgba(17, 197, 217, 0.4)' : '0 25px 50px -12px rgba(0, 0, 0, 0.5)', transition: 'all 0.3s ease' }}
                    >
                        {isEnhancing ? 'Enhancing...' : 'Enhance Model'}
                    </Button>
                </div>

                <div className="canvas-overlay-tr">
                    <button className="icon-btn">
                        <RotateCcw size={20} />
                    </button>
                    <button className="icon-btn">
                        <Maximize2 size={20} />
                    </button>
                </div>
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
                                <span style={{ fontSize: '0.75rem', fontFamily: 'var(--font-mono)', color: 'var(--color-soft-gold)' }}>0.3</span>
                            </div>
                            <div className="bar-container">
                                <div className="bar-fill" style={{ width: '30%' }} />
                            </div>

                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', paddingTop: '8px' }}>
                                <span style={{ fontSize: '0.875rem', color: 'white' }}>Metalness</span>
                                <span style={{ fontSize: '0.75rem', fontFamily: 'var(--font-mono)', color: 'var(--color-soft-gold)' }}>0.8</span>
                            </div>
                            <div className="bar-container">
                                <div className="bar-fill" style={{ width: '80%' }} />
                            </div>
                        </div>
                    </div>
                </div>

                <div className="panel-footer">
                    <Button variant="outline" style={{ width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}>
                        <Download size={18} />
                        Export Model
                    </Button>
                </div>
            </div>
        </div>
    );
};
