"""
Service wrapper for MiDaS / ZoeDepth.

Responsibilities:
- Load Depth estimation model
- Generate depth maps from RGB images
- Provide depth cues for the refinement module
"""

class DepthService:
    def estimate_depth(self, image_path: str):
        """
        Returns a depth map for the given image.
        """
        pass
