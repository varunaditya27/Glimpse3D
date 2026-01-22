
import asyncio
import logging
import gc
from typing import TypeVar, Callable, Any
from functools import wraps

logger = logging.getLogger(__name__)

# Try to import torch
try:
    import torch
except ImportError:
    torch = None

T = TypeVar("T")

def retry_gpu_operation(max_retries: int = 2, initial_delay: float = 1.0):
    """
    Decorator to retry async operations that might fail due to GPU OOM.
    Triggers VRAM cleanup between retries.
    """
    def decorator(func: Callable[..., Any]):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            delay = initial_delay
            last_error = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    error_str = str(e).lower()
                    
                    # Check for OOM or related GPU errors
                    is_oom = "out of memory" in error_str or "cuda error" in error_str or "alloc" in error_str
                    
                    if is_oom and attempt < max_retries:
                        logger.warning(f"GPU Error encountered in {func.__name__}: {e}. Retrying ({attempt+1}/{max_retries})...")
                        
                        # Cleanup VRAM
                        if torch and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                        
                        # Wait before retry
                        await asyncio.sleep(delay)
                        delay *= 2  # Exponential backoff
                    elif attempt < max_retries:
                         # Retry other errors if needed? For now, mainly GPU.
                         # But network flakes in download might also happen.
                         # Let's be conservative and only retry GPU OOM by default 
                         # UNLESS we want general robustness. 
                         # Let's retry generic errors once just in case.
                         logger.warning(f"Operation failed in {func.__name__}: {e}. Retrying ({attempt+1}/{max_retries})...")
                         await asyncio.sleep(delay)
                         delay *= 2
                    else:
                        logger.error(f"Operation {func.__name__} failed after {max_retries} retries.")
                        raise last_error
            
            raise last_error
        return wrapper
    return decorator
