import gc
import logging
import threading

logger = logging.getLogger(__name__)

# Mutex to ensure only one heavy model (Whisper vs LLM) utilizes the GPU at a time
gpu_lock = threading.Lock()

def is_cuda_available() -> bool:
    """Check if CUDA is truly available with PyTorch."""
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

def get_device_and_compute_type(preferred_device: str = "cuda", preferred_compute: str = "float16"):
    """
    Determine safe device and compute type based on actual hardware capability.
    Falls back gracefully to CPU and int8 if CUDA is not available.
    """
    cuda_ready = is_cuda_available()
    if preferred_device.lower() == "cuda" and cuda_ready:
        return "cuda", preferred_compute
    else:
        logger.info("Using CPU mode for execution (CUDA not available or disabled).")
        return "cpu", "int8"

def clean_memory():
    """Aggressively release memory and empty PyTorch CUDA cache."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            logger.info("PyTorch CUDA cache cleared.")
    except Exception:
        pass
