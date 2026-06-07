import { Suspense } from 'react';
import { Html } from '@react-three/drei';
import { ThreeCanvas } from './ThreeCanvas';
import { SplatViewer } from './SplatViewer';
import { ErrorBoundary } from './ErrorBoundary';

interface DualThreeCanvasProps {
    urlA: string;
    urlB: string;
}

// One half of the comparison: an independent R3F canvas hosting a single
// Gaussian splat. Each canvas has its own camera/controls so the two models
// can be inspected separately.
const SplatPane = ({ url, label }: { url: string; label: string }) => (
    <div style={{ flex: 1, position: 'relative', minWidth: 0, height: '100%' }}>
        <ThreeCanvas>
            <Suspense
                fallback={
                    <Html center>
                        <div style={{ color: 'white', whiteSpace: 'nowrap' }}>Loading 3D Model...</div>
                    </Html>
                }
            >
                {url ? (
                    <ErrorBoundary
                        fallback={
                            ((err: Error) => (
                                <Html center>
                                    <div
                                        style={{
                                            padding: '20px',
                                            color: '#ff4444',
                                            background: 'rgba(0,0,0,0.9)',
                                            borderRadius: '8px',
                                            border: '1px solid #ff4444',
                                        }}
                                    >
                                        <strong>Failed to load 3D Model</strong>
                                        <div style={{ fontSize: '12px', marginTop: '8px', whiteSpace: 'nowrap' }}>
                                            {err.message}
                                        </div>
                                    </div>
                                </Html>
                            )) as any
                        }
                    >
                        <SplatViewer url={url} />
                    </ErrorBoundary>
                ) : (
                    <Html center>
                        <div style={{ color: 'var(--color-fog-silver)', whiteSpace: 'nowrap' }}>
                            No model available
                        </div>
                    </Html>
                )}
            </Suspense>
        </ThreeCanvas>

        {/* Pane label */}
        <div
            style={{
                position: 'absolute',
                top: '16px',
                left: '16px',
                padding: '6px 12px',
                background: 'rgba(11, 12, 16, 0.8)',
                backdropFilter: 'blur(10px)',
                borderRadius: '8px',
                border: '1px solid var(--color-graphite-gray)',
                color: 'var(--color-pure-white)',
                fontSize: '0.875rem',
                fontWeight: 600,
                zIndex: 5,
            }}
        >
            {label}
        </div>
    </div>
);

// Side-by-side comparison of two Gaussian splats.
export const DualThreeCanvas = ({ urlA, urlB }: DualThreeCanvasProps) => {
    return (
        <div style={{ display: 'flex', width: '100%', height: '100%' }}>
            <SplatPane url={urlA} label="Model A" />
            <div style={{ width: '1px', background: 'var(--color-graphite-gray)' }} />
            <SplatPane url={urlB} label="Model B" />
        </div>
    );
};
