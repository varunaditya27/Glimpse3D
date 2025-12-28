# ✨ Glimpse3D Frontend

The premium, interactive frontend for the **Glimpse3D** system. Built with **React**, **Three.js**, and **Framer Motion** to deliver a "Modern Scientific Minimalism" experience.

## 🎨 Design Philosophy

**"Modern Scientific Minimalism"**
- **Aesthetic**: Sleek, futuristic, and high-contrast.
- **Colors**: Carbon Black (`#0B0C10`), Graphite Gray (`#1F2329`), Soft Gold (`#D4A857`), Ultramarine Cyan (`#11C5D9`).
- **Typography**: `Space Grotesk` (Headings) & `Manrope` (Body).
- **Atmosphere**: Cinematic noise overlays, glassmorphism, and fluid 3D backgrounds.

## 🛠 Tech Stack

- **Core**: React 18, TypeScript, Vite
- **3D Engine**: Three.js, React Three Fiber (`@react-three/fiber`), Drei (`@react-three/drei`)
- **Styling**: Vanilla CSS with CSS Variables (No Tailwind dependency)
- **Animations**: Framer Motion
- **Icons**: Lucide React

## 🚀 Features

### 1. **Immersive Landing Page**
- **Liquid 3D Hero**: A mesmerizing, distorting gold sphere with floating particles.
- **Cinematic Entrance**: Smooth fade-ins and parallax effects.
- **Interactive UI**: Hover-reactive buttons and text gradients.

### 2. **Professional Workspace**
- **3-Panel Layout**:
    - **Left Rail**: Toolset (Upload, Layers, Enhance, Settings).
    - **Center Canvas**: Real-time 3D viewer with orbit controls and environment lighting.
    - **Right Panel**: Model properties (Dimensions, Materials) with glassmorphism styling.
- **Enhancement Loop**: "Enhance" button with laser-scanning visualization.
- **Micro-interactions**: Glowing active states, spring animations, and smooth transitions.

## 📂 Directory Structure

```
frontend/
├── src/
│   ├── components/     # Reusable UI (Button, ThreeCanvas)
│   ├── pages/          # Route views (Landing, Workspace)
│   ├── styles/         # Global CSS & Variables (index.css)
│   ├── App.tsx         # Routing & Layout
│   └── main.tsx        # Entry point
├── public/             # Static assets
└── index.html          # HTML entry
```

## ⚡ Getting Started

1.  **Install Dependencies**
    ```bash
    npm install
    ```

2.  **Run Development Server**
    ```bash
    npm run dev
    ```

3.  **Build for Production**
    ```bash
    npm run build
    ```

## 🔗 Integration

This frontend connects to the **Glimpse3D Backend** (FastAPI) for:
- Image Upload
- 3D Generation (TripoSR/LGM)
- Model Enhancement (SDXL + ControlNet)
