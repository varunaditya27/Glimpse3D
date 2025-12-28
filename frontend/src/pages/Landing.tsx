import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { Button } from '../components/Button';
import { ThreeCanvas } from '../components/ThreeCanvas';
import { Sphere, MeshDistortMaterial, Float, Stars } from '@react-three/drei';

const HeroBackground = () => {
    return (
        <ThreeCanvas>
            <Float speed={2} rotationIntensity={1} floatIntensity={2}>
                <Sphere args={[1, 64, 64]} position={[0, 0, 0]} scale={2.2}>
                    <MeshDistortMaterial
                        color="#D4A857"
                        attach="material"
                        distort={0.6}
                        speed={3}
                        roughness={0.1}
                        metalness={0.9}
                        envMapIntensity={1.5}
                    />
                </Sphere>
            </Float>
            <Stars radius={100} depth={50} count={5000} factor={4} saturation={0} fade speed={1} />
        </ThreeCanvas>
    );
};

export const Landing = () => {
    const navigate = useNavigate();

    return (
        <div className="landing-container">
            {/* Background Layer */}
            <div style={{ position: 'absolute', inset: 0, zIndex: 0, opacity: 0.6 }}>
                <HeroBackground />
            </div>

            {/* Content Layer */}
            <div className="landing-content">
                <motion.div
                    initial={{ opacity: 0, y: 40, filter: 'blur(10px)' }}
                    animate={{ opacity: 1, y: 0, filter: 'blur(0px)' }}
                    transition={{ duration: 1, ease: [0.22, 1, 0.36, 1] }}
                >
                    <h1 className="landing-title">
                        <span style={{ color: 'var(--color-pure-white)', display: 'block' }}>Turn One Image Into</span>
                        <span className="text-gradient-gold" style={{ display: 'block', marginTop: '-10px' }}>3D Reality</span>
                    </h1>

                    <p className="landing-subtitle" style={{ animation: 'float 6s ease-in-out infinite' }}>
                        Glimpse3D transforms a single 2D photo into a high-quality, refined 3D model using advanced AI.
                    </p>

                    <div style={{ display: 'flex', gap: '20px', justifyContent: 'center', marginTop: '32px' }}>
                        <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                            <Button
                                size="lg"
                                onClick={() => navigate('/workspace')}
                                style={{ minWidth: '220px', boxShadow: '0 0 30px rgba(212, 168, 87, 0.3)' }}
                            >
                                Start Creating
                            </Button>
                        </motion.div>
                        <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                            <Button
                                variant="outline"
                                size="lg"
                                onClick={() => window.open('https://github.com/varunaditya27/Glimpse3D', '_blank')}
                                style={{ backdropFilter: 'blur(10px)' }}
                            >
                                View on GitHub
                            </Button>
                        </motion.div>
                    </div>
                </motion.div>
            </div>

            {/* Footer / Brand */}
            <div style={{ position: 'absolute', bottom: '40px', left: 0, right: 0, textAlign: 'center', color: 'var(--color-fog-silver)', fontSize: '0.875rem', letterSpacing: '0.1em', textTransform: 'uppercase', opacity: 0.7 }}>
                Glimpse3D AI System v1.0
            </div>
        </div>
    );
};
