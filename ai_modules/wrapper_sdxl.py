import torch
import numpy as np
from PIL import Image

class SDXLWrapper:
    """
    Wrapper for SDXL + ControlNet Image Enhancement.
    """
    def __init__(self, device="cuda"):
        self.device = device
        self.pipe = None

    def load_model(self):
        if self.pipe is None:
            try:
                from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
                
                # Load ControlNet
                controlnet = ControlNetModel.from_pretrained(
                    "diffusers/controlnet-canny-sdxl-1.0",
                    torch_dtype=torch.float16
                )
                
                # Load VAE
                vae = AutoencoderKL.from_pretrained(
                    "madebyollin/sdxl-vae-fp16-fix", 
                    torch_dtype=torch.float16
                )
                
                # Load Pipeline
                self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                    "stabilityai/stable-diffusion-xl-base-1.0",
                    controlnet=controlnet,
                    vae=vae,
                    torch_dtype=torch.float16
                ).to(self.device)
                
            except ImportError:
                print("Diffusers not installed. Running in Mock Mode.")
            except Exception as e:
                print(f"Failed to load SDXL: {e}")

    def enhance(self, image: np.ndarray, prompt: str = "high quality, detailed, photorealistic", strength: float = 0.5):
        """
        Enhance an image using SDXL.
        
        Args:
            image: (H, W, 3) numpy array [0-255]
            
        Returns:
            enhanced_image: (H, W, 3) numpy array
        """
        if self.pipe is None:
            self.load_model()
            if self.pipe is None:
                # Mock: Return original image with slight noise
                return image

        # Convert to PIL
        pil_image = Image.fromarray(image.astype(np.uint8))
        
        # Prepare ControlNet input (Canny edge)
        import cv2
        image_cv = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_canny = cv2.Canny(image_cv, 100, 200)
        image_canny = image_canny[:, :, None]
        image_canny = np.concatenate([image_canny, image_canny, image_canny], axis=2)
        control_image = Image.fromarray(image_canny)

        with torch.no_grad():
            output = self.pipe(
                prompt,
                image=pil_image,  # img2img input
                control_image=control_image, # ControlNet input
                strength=strength,
                num_inference_steps=20,
                guidance_scale=7.5
            ).images[0]
            
        return np.array(output)
