"""
Service wrapper for SyncDreamer multi-view generation.

Responsibilities:
- Load SyncDreamer model
- Generate consistent multi-view images from single input
- Provide views for depth estimation and refinement
"""

import logging
import os
import sys
from typing import Dict, Any
from pathlib import Path

# Add ai_modules to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from ai_modules.sync_dreamer.inference import generate_multiview

logger = logging.getLogger(__name__)

class SyncDreamerService:
    def __init__(self):
        self.logger = logger

    async def generate_views(self, image_path: str, output_dir: str) -> Dict[str, Any]:
        """
        Generates novel views using SyncDreamer.
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save generated views
            
        Returns:
            Dict with 'success', 'view_paths' (dict mapping index/name to path)
        """
        try:
            self.logger.info(f"Generating views for {image_path} using SyncDreamer")
            
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Use the synchronous generation function (SyncDreamer uses GPU heavily, blocks)
            # In a real async server, this should maybe run in an executor, but for now strict call is fine.
            paths_list = generate_multiview(image_path, str(output_path), elevation=30.0)
            
            if not paths_list:
                 return {
                    'success': False,
                    'error': 'No views generated',
                    'view_paths': {}
                }

            # Convert list to dict for pipeline consistency (view_00, view_01...)
            view_paths = {}
            for i, path in enumerate(paths_list):
                 view_name = f"view_{i:02d}"
                 view_paths[view_name] = path

            return {
                'success': True,
                'view_paths': view_paths
            }

        except Exception as e:
            error_msg = f"SyncDreamer generation failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {
                'success': False,
                'error': error_msg,
                'view_paths': {}
            }
