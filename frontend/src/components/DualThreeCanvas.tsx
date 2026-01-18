import { useEffect, useRef } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Environment, PerspectiveCamera } from '@react-three/drei';
import { motion } from 'framer-motion';

interface DualThreeCanvasProps {
    leftModelUrl: string | null;
    rightModelUrl: string | null;
    syncCameras?: boolean;
    leftTitle?: string;
    rightTitle?: string;
}

// Placeholder mesh component (replace with actual model loader)
const ModelMesh = ({ color }: { color: string }) => {
    return (
        <mesh>
            <torusKnotGeometry args={[0.45, 0.15, 128, 32]} />
            <meshStandardMaterial
                color={color}
                roughness={0.2}
                metalness={0.9}
                emissive={color}
                emissiveIntensity={0.2}
            />
        </mesh>
    );
};

const SceneContent = ({ modelUrl, color }: { modelUrl: string | null; color: string }) => {
    // TODO: Load actual .ply or .glb model from modelUrl
    // For now, showing placeholder geometry
    return (
        <>
            <PerspectiveCamera makeDefault position={[0, 0, 3]} />
            <OrbitControls enableDamping dampingFactor={0.05} />
            <Environment preset="city" />
            <ambientLight intensity={0.5} />
            <directionalLight position={[5, 5, 5]} intensity={1} />
            <directionalLight position={[-5, -5, -5]} intensity={0.5} />
            {modelUrl && <ModelMesh color={color} />}
        </>
    );
};

export const DualThreeCanvas = ({
    leftModelUrl,
    rightModelUrl,
    syncCameras = false,
    leftTitle = "Model A",
    rightTitle = "Model B"
}: DualThreeCanvasProps) => {
    return (
        <div style={{ display: 'flex', width: '100%', height: '100%', position: 'relative' }}>
            {/* Left Canvas */}
            <div style={{ flex: 1, position: 'relative', borderRight: '1px solid var(--color-graphite-gray)' }}>
                <Canvas
                    style={{ background: 'var(--color-carbon-black)' }}
                    gl={{ antialias: true, alpha: true }}
                >
                    <SceneContent modelUrl={leftModelUrl} color="#D4A857" />
                </Canvas>
                
                {/* Left Label */}
                <div
                    style={{
                        position: 'absolute',
                        top: '16px',
                        left: '16px',
                        padding: '8px 16px',
                        background: 'rgba(11, 12, 16, 0.8)',
                        backdropFilter: 'blur(10px)',
                        borderRadius: '8px',
                        border: '1px solid rgba(212, 168, 87, 0.3)',
                        zIndex: 10
                    }}
                >
                    <span style={{ color: 'var(--color-soft-gold)', fontSize: '0.875rem', fontWeight: 600 }}>
                        {leftTitle}
                    </span>
                </div>
            </div>

            {/* Center Divider with Glassmorphism */}
            <motion.div
                initial={{ scaleY: 0 }}
                animate={{ scaleY: 1 }}
                transition={{ duration: 0.5, ease: "easeOut" }}
                style={{
                    width: '2px',
                    background: 'linear-gradient(to bottom, transparent, var(--color-soft-gold), transparent)',
                    position: 'absolute',
                    left: '50%',
                    top: '10%',
                    bottom: '10%',
                    transform: 'translateX(-50%)',
                    boxShadow: '0 0 20px rgba(212, 168, 87, 0.5)',
                    zIndex: 20
                }}
            />

            {/* Right Canvas */}
            <div style={{ flex: 1, position: 'relative' }}>
                <Canvas
                    style={{ background: 'var(--color-carbon-black)' }}
                    gl={{ antialias: true, alpha: true }}
                >
                    <SceneContent modelUrl={rightModelUrl} color="#11C5D9" />
                </Canvas>
                
                {/* Right Label */}
                <div
                    style={{
                        position: 'absolute',
                        top: '16px',
                        right: '16px',
                        padding: '8px 16px',
                        background: 'rgba(11, 12, 16, 0.8)',
                        backdropFilter: 'blur(10px)',
                        borderRadius: '8px',
                        border: '1px solid rgba(17, 197, 217, 0.3)',
                        zIndex: 10
                    }}
                >
                    <span style={{ color: 'var(--color-ultramarine-cyan)', fontSize: '0.875rem', fontWeight: 600 }}>
                        {rightTitle}
                    </span>
                </div>
            </div>

            {/* Sync Cameras Toggle (if needed) */}
            {syncCameras && (
                <div
                    style={{
                        position: 'absolute',
                        bottom: '16px',
                        left: '50%',
                        transform: 'translateX(-50%)',
                        padding: '6px 12px',
                        background: 'rgba(11, 12, 16, 0.9)',
                        backdropFilter: 'blur(10px)',
                        borderRadius: '20px',
                        border: '1px solid var(--color-graphite-gray)',
                        fontSize: '0.75rem',
                        color: 'var(--color-fog-silver)',
                        zIndex: 10
                    }}
                >
                    🔗 Cameras Synced
                </div>
            )}
        </div>
    );
};
