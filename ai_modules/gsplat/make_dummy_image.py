from PIL import Image
import numpy as np
import os

def create_dummy_image(path):
    # Create 512x512 red square with alpha
    img = Image.new('RGBA', (512, 512), (255, 0, 0, 255))
    img.save(path)
    print(f"Created dummy image at {path}")

if __name__ == "__main__":
    create_dummy_image("dummy_input.png")
