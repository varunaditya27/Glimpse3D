import { Canvas } from '@react-three/fiber';
import { OrbitControls, Environment, ContactShadows } from '@react-three/drei';
import { Suspense } from 'react';

interface ThreeCanvasProps {
    children?: React.ReactNode;
    autoRotate?: boolean;
}

export const ThreeCanvas = ({ children, autoRotate = false }: ThreeCanvasProps) => {
    return (
        <div style={{ width: '100%', height: '100%', position: 'relative', background: 'linear-gradient(to bottom, #0B0C10, #161A1F)' }}>
            <Canvas
                camera={{ position: [0, 2, 5], fov: 45 }}
                gl={{ antialias: true, alpha: true }}
                dpr={[1, 2]}
            >
                <Suspense fallback={null}>
                    <Environment preset="city" />
                    <ambientLight intensity={0.5} />
                    <spotLight position={[10, 10, 10]} angle={0.15} penumbra={1} intensity={1} castShadow />

                    {children}

                    <ContactShadows position={[0, -1, 0]} opacity={0.4} scale={10} blur={2.5} far={4} />
                    <OrbitControls
                        makeDefault
                        enablePan={false}
                        minPolarAngle={0}
                        maxPolarAngle={Math.PI / 1.5}
                        autoRotate={autoRotate}
                        autoRotateSpeed={1.0}
                    />
                </Suspense>
            </Canvas>

            {/* Vignette Overlay */}
            <div style={{ position: 'absolute', inset: 0, pointerEvents: 'none', background: 'radial-gradient(circle at center, transparent 0%, rgba(11, 12, 16, 0.6) 100%)' }} />
        </div>
    );
};
