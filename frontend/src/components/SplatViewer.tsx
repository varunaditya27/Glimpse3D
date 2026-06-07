import { useRef, useEffect } from 'react';
import { useThree, useFrame } from '@react-three/fiber';
import { Html } from '@react-three/drei';
import * as GaussianSplats3D from '@mkkellogg/gaussian-splats-3d';

interface SplatViewerProps {
    url: string;
}

// Reusable single Gaussian-splat renderer.
// Mirrors the ModelViewer logic from Workspace.tsx: it attaches a
// @mkkellogg/gaussian-splats-3d Viewer to the active R3F renderer/camera
// in non-self-driven mode and pumps update()/render() inside useFrame.
export const SplatViewer = ({ url }: SplatViewerProps) => {
    const { gl, camera } = useThree();
    const viewerRef = useRef<any>(null);

    useEffect(() => {
        if (!url) return;

        let viewer: any = null;

        const initViewer = async () => {
            try {
                // Ensure URL is safe (normalize Windows-style separators).
                const safeUrl = url.replace(/\\/g, '/');

                // @mkkellogg/gaussian-splats-3d v0.4.7+
                // selfDrivenMode: false  -> we drive the loop via useFrame
                // useBuiltInControls: false -> R3F OrbitControls owns the camera
                const ViewerClass = (GaussianSplats3D as any).Viewer;

                viewer = new ViewerClass({
                    selfDrivenMode: false,
                    useBuiltInControls: false,
                    rootElement: gl.domElement.parentElement,
                    renderer: gl,
                    camera: camera,
                    gpuAcceleratedSort: true,
                });

                await viewer.addSplatScene(safeUrl, {
                    showLoadingUI: false,
                    position: [0, 0, 0],
                    rotation: [0, 0, 0, 1],
                    scale: [1.5, 1.5, 1.5],
                    splatAlphaRemovalThreshold: 5,
                });

                viewerRef.current = viewer;
            } catch (err) {
                console.error('GSplat Viewer Init Error:', err);
            }
        };

        initViewer();

        return () => {
            viewerRef.current = null;
            if (viewer) {
                viewer.dispose();
            }
        };
    }, [url, gl, camera]);

    // Pump the viewer each frame.
    useFrame(() => {
        if (viewerRef.current) {
            viewerRef.current.update();
            viewerRef.current.render();
        }
    });

    // Render nothing meaningful for an empty url; show a small loading hint
    // until the viewer is ready.
    if (!url) {
        return null;
    }

    return (
        <group>
            <Html position={[0, 2, 0]} center>
                {viewerRef.current ? null : (
                    <div className="text-white text-sm bg-black/50 px-2 py-1 rounded">
                        Loading Splats...
                    </div>
                )}
            </Html>
        </group>
    );
};
