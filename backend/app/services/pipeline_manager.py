"""
Orchestrator for the entire 3D generation pipeline.

Responsibilities:
- Manage state transitions (Uploaded -> Coarse -> Refined)
- Coordinate calls between different AI services (Zero123, Depth, Diffusion, GSplat)
- Handle error recovery and logging
"""

class PipelineManager:
    def __init__(self):
        pass

    async def run_pipeline(self, image_path: str):
        """
        Executes the full generation pipeline.
        """
        pass
