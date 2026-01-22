import { useState, useRef, Suspense, useEffect } from 'react';
import { ThreeCanvas } from '../components/ThreeCanvas';
import { ErrorBoundary } from '../components/ErrorBoundary';
import { Button } from '../components/Button';
import { Upload, Sliders, Sparkles, Download, Maximize2, RotateCcw, Wand2, Folder, X, AlertTriangle, CheckCircle, Info } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import clsx from 'clsx';
import { Html } from '@react-three/drei';
import { useThree, useFrame } from '@react-three/fiber';
import * as GaussianSplats3D from '@mkkellogg/gaussian-splats-3d';

// --- Toast System ---
type ToastType = 'success' | 'error' | 'info' | 'warning';
interface ToastMsg {
    id: string;
    type: ToastType;
    message: string;
}

const ToastItem = ({ toast, onDismiss }: { toast: ToastMsg; onDismiss: (id: string) => void }) => {
    return (
        <motion.div
            initial={{ opacity: 0, y: 50, scale: 0.9 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9, transition: { duration: 0.2 } }}
            style={{
                background: 'rgba(22, 27, 34, 0.95)',
                backdropFilter: 'blur(10px)',
                border: `1px solid ${toast.type === 'error' ? '#ff4444' : toast.type === 'warning' ? '#ffbb33' : toast.type === 'success' ? '#00C851' : '#33b5e5'}`,
                borderRadius: '8px',
                padding: '12px 16px',
                marginBottom: '8px',
                display: 'flex',
                alignItems: 'center',
                gap: '12px',
                color: 'white',
                fontSize: '0.875rem',
                boxShadow: '0 4px 12px rgba(0,0,0,0.5)',
                width: '320px',
                pointerEvents: 'auto'
            }}
        >
            {toast.type === 'error' && <AlertTriangle size={18} color="#ff4444" />}
            {toast.type === 'warning' && <AlertTriangle size={18} color="#ffbb33" />}
            {toast.type === 'success' && <CheckCircle size={18} color="#00C851" />}
            {toast.type === 'info' && <Info size={18} color="#33b5e5" />}

            <div style={{ flex: 1 }}>{toast.message}</div>

            <button onClick={() => onDismiss(toast.id)} style={{ background: 'none', border: 'none', color: '#aaa', cursor: 'pointer' }}>
                <X size={14} />
            </button>
        </motion.div>
    );
};
// --------------------

// Re-implementing ModelViewer with the actual library logic
// We need to access useThree

const ModelViewer = ({ url }: { url: string, material: any }) => {
    const { gl, camera } = useThree();
    const viewerRef = useRef<any>(null);

    useEffect(() => {
        let viewer: any = null;

        const initViewer = async () => {
            try {
                // Ensure URL is safe
                const safeUrl = url.replace(/\\/g, '/');

                // Initialize the viewer
                // @mkkellogg/gaussian-splats-3d v0.4.7+
                // We configure it to NOT drive the loop ('selfDrivenMode': false)
                // and NOT handle controls ('useBuiltInControls': false)
                const ViewerClass = (GaussianSplats3D as any).Viewer;

                viewer = new ViewerClass({
                    selfDrivenMode: false,
                    useBuiltInControls: false,
                    rootElement: gl.domElement.parentElement, // or just null if we don't want it efficiently attaching events
                    renderer: gl,
                    camera: camera,
                    gpuAcceleratedSort: true
                });

                // Load the file
                // addSplatScene(path, options)
                await viewer.addSplatScene(safeUrl, {
                    showLoadingUI: false,
                    position: [0, 0, 0],
                    rotation: [0, 0, 0, 1],
                    scale: [1.5, 1.5, 1.5],
                    splatAlphaRemovalThreshold: 5, // Remove very transparent splats
                });

                viewerRef.current = viewer;

            } catch (err) {
                console.error("GSplat Viewer Init Error:", err);
            }
        };

        initViewer();

        return () => {
            if (viewer) {
                viewer.dispose();
            }
        };
    }, [url, gl, camera]);

    // Hook into the render loop to update the viewer
    useFrame(() => {
        if (viewerRef.current) {
            viewerRef.current.update();
            viewerRef.current.render();
        }
    });

    return (
        <group>
            {/* 
               The viewer renders directly to the WebGL context usually.
               We might need to ensure it plays nice with other objects.
               However, strictly speaking, @mkkellogg's viewer often wants to be the main thing.
               If 'selfDrivenMode: false', we call render().
               Note: 'viewer.render()' forces a render command. 
               This might conflict with R3F's auto-render if not careful.
               But usually it just draws splats.
               
               Wait, if we use R3F, R3F clears and renders the scene.
               If we call viewer.render() inside useFrame, it draws on top (or as part of it).
               The library creates a Mesh and adds it to the scene usually?
               Actually, checking docs (from memory/search):
               It can act as a scene object if we use `viewer.getSplatMesh()`?
               
               Let's try a safer approach:
               It generally adds a mesh to the scene if we pass 'scene' in options?
               Let's pass 'scene' to constructor if supported, or check if we can get the mesh.
             */}
            <mesh position={[0, -2, 0]} rotation={[-Math.PI / 2, 0, 0]}>
                <planeGeometry args={[10, 10]} />
                <shadowMaterial transparent opacity={0.3} />
            </mesh>

            <Html position={[0, 2, 0]} center>
                {viewerRef.current ? null : <div className="text-white text-sm bg-black/50 px-2 py-1 rounded">Loading Splats...</div>}
            </Html>
        </group>
    );
};

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
    // Tools: 'project', 'editor', 'ai', 'export'
    // Tools: 'project', 'editor', 'ai', 'export'
    const [activeTool, setActiveTool] = useState('project');
    const [status, setStatus] = useState<'idle' | 'uploading' | 'processing' | 'ready' | 'enhancing' | 'enhanced'>('idle');
    const [progress, setProgress] = useState(0);
    const [statusText, setStatusText] = useState('');
    const [resultUrl, setResultUrl] = useState<string | null>(null);

    // Material State
    const [material, setMaterial] = useState({
        color: '#D4A857',
        roughness: 0.2,
        metalness: 0.9,
        emissive: 0,
        wireframe: false,
        autoRotate: false
    });

    // Metadata State
    const [metadata, setMetadata] = useState<{ vertex_count: number; format: string; file_size_human: string } | null>(null);
    const [exportFormat, setExportFormat] = useState('glb');
    const [jobId, setJobId] = useState<string | null>(null);

    // AI Command Bar State
    const [prompt, setPrompt] = useState('');
    const [isGenerating, setIsGenerating] = useState(false);

    // Toast State
    const [toasts, setToasts] = useState<ToastMsg[]>([]);

    const addToast = (message: string, type: ToastType = 'info') => {
        const id = Math.random().toString(36).substring(7);
        setToasts(prev => [...prev, { id, type, message }]);
        setTimeout(() => removeToast(id), 5000); // Auto dismiss
    };

    const removeToast = (id: string) => {
        setToasts(prev => prev.filter(t => t.id !== id));
    };

    // Using a ref properly
    const inputRef = useRef<HTMLInputElement>(null);

    const handleUploadClick = () => {
        inputRef.current?.click();
    };

    const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
        if (!e.target.files || e.target.files.length === 0) return;

        const file = e.target.files[0];
        setStatus('uploading');
        setProgress(0);
        setStatusText('Uploading image...');

        const formData = new FormData();
        formData.append('file', file);

        try {
            // 1. Upload
            const uploadRes = await fetch('/upload/', {
                method: 'POST',
                body: formData,
            });
            const uploadData = await uploadRes.json();

            if (!uploadData.success) {
                throw new Error(uploadData.error || 'Upload failed');
            }

            handleUpload(file);
        } catch (error: any) {
            console.error(error);
            setStatus('idle');
            setStatusText('Error: ' + String(error.message));
            addToast(error.message || "Upload Failed", 'error');
        }
    };

    const fetchMetadata = async (id: string) => {
        try {
            const res = await fetch(`/export/metadata/${id}`);
            const data = await res.json();
            setMetadata(data);
        } catch (e) {
            console.error("Failed to fetch metadata", e);
        }
    };

    // WebSocket Connection
    useEffect(() => {
        // Connect to WebSocket
        const clientId = `client_${Math.floor(Math.random() * 1000)}`;
        const wsUrl = `${(import.meta.env.VITE_API_URL || 'http://localhost:8000').replace('http', 'ws')}/ws/${clientId}`;

        console.log("Connecting to WebSocket:", wsUrl);
        const ws = new WebSocket(wsUrl);

        ws.onopen = () => {
            console.log("WebSocket connected");
            // addToast("Connected to Server", 'success'); 
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                console.log("WS Message:", data);

                // Filter for our current job if strictly needed, 
                // but for now we consume all updates (single user assumption or "active job" logic)
                // If data.job_id matches our current jobId (if set) OR if we are just waiting for any job we started.

                if (data.type === 'progress_update') {
                    // Only update if it matches our job OR if we are in a state where we accept updates
                    // Ideally check data.job_id === jobId
                    // But jobId might be state variable.

                    if (data.status) setStatus(data.status);
                    if (data.progress !== undefined) setProgress(data.progress);
                    if (data.message) setStatusText(data.message);

                    if (data.warnings && data.warnings.length > 0) {
                        data.warnings.forEach((w: string) => addToast(w, 'warning'));
                    }
                }
                else if (data.type === 'job_completed') {
                    if (data.model_url) {
                        setResultUrl(data.model_url); // This is absolute path in backend logic?
                        // Backend sends /outputs/... relative to domain root
                        // setStatus('completed'); // 'enhanced' is usually the final state for UI visibility?
                        // Let's stick to status from backend 'completed' mapping to our UI 'enhanced' or 'viewer'
                        setStatus('enhanced');
                        setStatusText('Generation complete');
                        setProgress(1.0);
                        setIsGenerating(false);

                        // Fetch Metadata if available
                        if (data.job_id) fetchMetadata(data.job_id);

                        addToast("Model Generation Complete!", 'success');
                    }
                }
                else if (data.type === 'job_failed') {
                    setStatus('idle');
                    setStatusText(`Failed: ${data.error}`);
                    // alert(`Generation Failed: ${data.error}`); // Use Toast instead
                    addToast(`Generation Failed: ${data.error}`, 'error');
                    setIsGenerating(false);
                }

            } catch (e) {
                console.error("Error parsing WS message:", e);
            }
        };

        ws.onerror = (e) => {
            console.error("WebSocket error:", e);
            // addToast("Connection Error", 'error');
        };

        ws.onclose = () => {
            console.log("WebSocket disconnected");
        };

        return () => {
            ws.close();
        };
    }, []); // Run once on mount

    // Poll status (Legacy / Fallback removed)
    // const pollStatus = ... (REMOVED)

    // Instead of pollStatus, handleUpload directly sets job ID
    // and we wait for WS updates.

    const handleUpload = async (file: File) => {
        setStatus('uploading');
        setProgress(0);
        setStatusText('Uploading status...');

        const formData = new FormData();
        formData.append('file', file);

        const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

        try {
            // 1. Upload
            const uploadRes = await fetch(`${API_URL}/upload`, { method: 'POST', body: formData });
            if (!uploadRes.ok) throw new Error('Upload failed');
            const uploadData = await uploadRes.json();
            const imagePath = uploadData.file_path; // Backend absolute path

            // 2. Generate
            setStatus('processing');
            setStatusText('Starting generation...');

            const genRes = await fetch(`${API_URL}/generate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image_path: imagePath })
            });

            if (!genRes.ok) throw new Error('Generation start failed');
            const genData = await genRes.json();

            setJobId(genData.job_id);
            console.log("Job started:", genData.job_id);

            // Revert pollStatus call - We now rely on WebSocket
            // pollStatus(genData.job_id); 

        } catch (e: any) {
            console.error("Polling error", e);
            addToast(e.message || "Failed to start generation", 'error');
        }
    };

    // Camera Tracking Ref
    const cameraPoseRef = useRef({ azimuth: 0, elevation: 0, radius: 3 });

    const CameraTracker = () => {
        const { camera } = useThree();
        useFrame(() => {
            if (camera) {
                // Convert Cartesian to Spherical
                const x = camera.position.x;
                const y = camera.position.y;
                const z = camera.position.z;

                const radius = Math.sqrt(x * x + y * y + z * z);
                // Elevation (phi): angle from Y axis (or pitch). 
                // Typically: y = r * sin(elev) => elev = asin(y/r)
                // However, render_view.py convention seems to be: 
                // x = r * cos(elev) * sin(azi)
                // y = r * sin(elev)
                // z = r * cos(elev) * cos(azi)
                // This implies Y-up.

                const elevation = Math.asin(y / radius);
                const azimuth = Math.atan2(x, z); // Note: atan2(x, z) matches sin(azi)/cos(azi) = x/z = tan(azi)

                // Convert to degrees
                cameraPoseRef.current = {
                    radius: radius,
                    elevation: (elevation * 180) / Math.PI,
                    azimuth: (azimuth * 180) / Math.PI
                };
            }
        });
        return null;
    };


    const handleEnhance = async () => {
        if (!jobId) return;

        setStatus('enhancing');
        setStatusText('Refining detail based on current view...');

        try {
            // Use camera params
            const { azimuth, elevation, radius } = cameraPoseRef.current;
            console.log(`Enhancing View: Azi=${azimuth.toFixed(1)}, Elev=${elevation.toFixed(1)}, Rad=${radius.toFixed(1)}`);

            const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

            const res = await fetch(`${API_URL}/refine`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model_id: jobId,
                    prompt: prompt || "High quality, detailed, 4k texture",
                    intensity: 0.5,
                    iterations: 100,
                    view_azimuth: azimuth,
                    view_elevation: elevation,
                    view_radius: radius
                })
            });

            if (!res.ok) {
                // If 404/500, we catch it
                const err = await res.json();
                throw new Error(err.detail || 'Refinement request failed');
            }

            // Backend should return quickly or we wait for WS updates?
            // The Refine Endpoint is async but awaits the process? 
            // Looking at refine.py, it awaits the whole process.
            // That might timeout HTTP. Ideally refine endpoint should return a Task ID.
            // But for this MVP, let's assume it waits (might take 30s-1m).
            // If it waits, we get the result here.

            const data = await res.json();

            if (data.status === 'completed') {
                setStatus('enhanced');
                setStatusText('Enhancement Complete!');
                setResultUrl(data.updated_model_url);
                // We could show comparison images too if we had UI for it
                setIsGenerating(false);
                addToast("Enhancement Successful!", 'success');
            }

        } catch (e: any) {
            console.error("Enhance error:", e);
            setStatus('ready'); // Revert
            setStatusText('Enhancement failed: ' + e.message);
            addToast(e.message || "Enhancement failed", 'error');
            setIsGenerating(false);
        }
    };

    const handleDownload = () => {
        if (!jobId) return;
        const handleDownload = async () => {
            if (!jobId) return;

            try {
                setStatusText('Downloading model...');
                // Trigger download via link to avoid blocking UI with large blob fetch if unnecessary, 
                // but for proper error handling we might want fetch.
                // Let's use window.location for simplicity if it works, but fetch allows us to catch 404/500 better.

                const response = await fetch(`${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/export/${jobId}?format=${exportFormat}`);

                if (!response.ok) {
                    const err = await response.json();
                    throw new Error(err.detail || 'Download failed');
                }

                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `model_${jobId}.${exportFormat}`;
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);

                setStatusText('Download complete');
                setTimeout(() => setStatusText(''), 3000);
                addToast("Download Started", 'success');

            } catch (e: any) {
                console.error("Download error:", e);
                // alert(`Download failed: ${e.message}`);
                addToast(`Download failed: ${e.message}`, 'error');
                setStatusText('Download failed');
            }
        };
        handleDownload(); // Call inner async
    };

    const handleReset = () => {
        setStatus('idle');
        setProgress(0);
        setStatusText('');
        setResultUrl(null);
        setJobId(null);
        setMetadata(null);
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
        if (inputRef.current) inputRef.current.value = '';
    };

    const handlePromptSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        alert("Text-to-texture editing coming soon!");
    };


    return (
        <ErrorBoundary fallback={(err: Error) => (
            <div style={{ padding: '40px', color: 'red', background: '#000', height: '100vh', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
                <h1>CRITICAL UI CRASH</h1>
                <pre style={{ fontSize: '16px', background: '#222', padding: '20px', borderRadius: '10px' }}>{err.message}</pre>
                <button onClick={() => window.location.reload()} style={{ padding: '10px 20px', marginTop: '20px', fontSize: '16px' }}>Reload App</button>
            </div>
        ) as any}>
            <div className="workspace-container">

                {/* Toast Container */}
                <div style={{ position: 'absolute', bottom: '24px', right: '24px', zIndex: 9999, display: 'flex', flexDirection: 'column', alignItems: 'flex-end', pointerEvents: 'none' }}>
                    <AnimatePresence>
                        {toasts.map(toast => (
                            <ToastItem key={toast.id} toast={toast} onDismiss={removeToast} />
                        ))}
                    </AnimatePresence>
                </div>

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

                <div style={{ position: 'absolute', top: 10, left: 80, zIndex: 9999, background: 'rgba(0,0,0,0.8)', color: '#0f0', padding: '10px', fontSize: '12px', fontFamily: 'monospace', pointerEvents: 'none' }}>
                    <div>STATUS: {status}</div>
                    <div>URL: {resultUrl || 'null'}</div>
                    <div>JOB: {jobId || 'null'}</div>
                    <div>ERR: {statusText || 'none'}</div>
                    <div>PROG: {progress}%</div>
                </div>

                {/* Center - Canvas */}
                <div className="canvas-area">
                    <ThreeCanvas autoRotate={material.autoRotate}>
                        <CameraTracker />
                        <Suspense fallback={
                            <Html center>
                                <div style={{ color: 'white', whiteSpace: 'nowrap' }}>
                                    Loading 3D Model... ({resultUrl})
                                </div>
                            </Html>
                        }>
                            {/* 3D Content based on state */}
                            {resultUrl ? (
                                <ErrorBoundary fallback={((err: Error) => (
                                    <Html center>
                                        <div style={{ padding: '20px', color: '#ff4444', background: 'rgba(0,0,0,0.9)', borderRadius: '8px', border: '1px solid #ff4444' }}>
                                            <strong>Failed to load 3D Model</strong>
                                            <div style={{ fontSize: '12px', marginTop: '8px', whiteSpace: 'nowrap' }}>{err.message}</div>
                                        </div>
                                    </Html>
                                )) as any}>
                                    <ModelViewer url={resultUrl} material={material} />
                                </ErrorBoundary>
                            ) : (status === 'idle' || status === 'uploading' ? null : (
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
                            ))}
                        </Suspense>
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
                                <Button onClick={handleUploadClick}>Select File</Button>
                                <input
                                    ref={inputRef}
                                    type="file"
                                    accept="image/png, image/jpeg"
                                    onChange={handleFileChange}
                                    style={{ display: 'none' }}
                                />
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
                                        <span style={{ fontSize: '0.875rem', fontFamily: 'var(--font-mono)' }}>{metadata ? metadata.vertex_count.toLocaleString() : '-'}</span>
                                    </div>
                                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                        <span style={{ fontSize: '0.875rem', color: 'var(--color-fog-silver)' }}>Format</span>
                                        <span style={{ fontSize: '0.875rem', fontFamily: 'var(--font-mono)' }}>{metadata ? metadata.format : '-'}</span>
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
                            <div className="prop-group">
                                <label className="prop-label">Export Options</label>
                                <div className="prop-card">
                                    <div style={{ marginBottom: '12px' }}>
                                        <label style={{ display: 'block', fontSize: '0.875rem', color: 'white', marginBottom: '8px' }}>Format</label>
                                        <select
                                            value={exportFormat}
                                            onChange={(e) => setExportFormat(e.target.value)}
                                            style={{ width: '100%', padding: '8px', borderRadius: '8px', background: 'rgba(0,0,0,0.3)', border: '1px solid var(--color-graphite-gray)', color: 'white' }}
                                        >
                                            <option value="glb">GLB (Recommended)</option>
                                            <option value="obj">OBJ</option>
                                            <option value="ply">PLY</option>
                                        </select>
                                    </div>
                                    <Button
                                        variant="outline"
                                        disabled={status !== 'enhanced' || !jobId}
                                        onClick={handleDownload}
                                        style={{ width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}
                                    >
                                        <Download size={18} />
                                        Download Model
                                    </Button>
                                </div>
                            </div>
                        )}

                    </div>
                </div>
            </div>
        </ErrorBoundary>
    );
};
