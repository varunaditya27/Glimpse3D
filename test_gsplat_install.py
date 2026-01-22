import torch
import sys

try:
    import gsplat
    print(f"gsplat imported successfully!")
    print(f"Version: {getattr(gsplat, '__version__', 'Unknown')}")
    
    if hasattr(gsplat, "rasterization"):
        print("gsplat.rasterization found.")
        print(gsplat.rasterization.__doc__)
    else:
        print("gsplat.rasterization NOT found. Available attributes:")
        print(dir(gsplat))
        
except ImportError as e:
    print(f"Failed to import gsplat: {e}")
    sys.exit(1)
except Exception as e:
    print(f"An error occurred: {e}")
    sys.exit(1)
