"""
Array backend abstraction for NumPy/CuPy
Enables CUDA acceleration when CuPy is available, falls back to NumPy otherwise
"""
import sys

# Try to import CuPy
try:
    import cupy as cp
    HAS_CUPY = True
    CUDA_AVAILABLE = cp.cuda.is_available()
except ImportError:
    HAS_CUPY = False
    CUDA_AVAILABLE = False
    cp = None

import numpy as np


class Backend:
    """Array backend that supports both NumPy and CuPy"""
    
    def __init__(self, use_cuda=None):
        """
        Initialize backend
        
        Args:
            use_cuda: True to force CUDA, False to force CPU, None for auto-detect
        """
        if use_cuda is None:
            self.use_cuda = CUDA_AVAILABLE
        else:
            if use_cuda and not CUDA_AVAILABLE:
                print("WARNING: CUDA requested but CuPy not available or no CUDA device found. Falling back to NumPy.")
                self.use_cuda = False
            else:
                self.use_cuda = use_cuda
        
        if self.use_cuda:
            self.xp = cp
            self.device = 'cuda'
            print(f"✓ Using CuPy backend (CUDA) - Device: {cp.cuda.Device().compute_capability}")
        else:
            self.xp = np
            self.device = 'cpu'
            print("✓ Using NumPy backend (CPU)")
    
    def array(self, data, dtype=None):
        """Create array on current device"""
        if dtype is not None:
            return self.xp.array(data, dtype=dtype)
        return self.xp.array(data)
    
    def zeros(self, shape, dtype=np.float64):
        """Create zeros array"""
        return self.xp.zeros(shape, dtype=dtype)
    
    def ones(self, shape, dtype=np.float64):
        """Create ones array"""
        return self.xp.ones(shape, dtype=dtype)
    
    def random_randn(self, *shape):
        """Random normal distribution"""
        return self.xp.random.randn(*shape)
    
    def to_cpu(self, array):
        """Move array to CPU (convert CuPy to NumPy)"""
        if self.use_cuda and isinstance(array, cp.ndarray):
            return cp.asnumpy(array)
        return array
    
    def to_device(self, array):
        """Move array to current device"""
        if self.use_cuda:
            if isinstance(array, np.ndarray):
                return cp.asarray(array)
            return array
        else:
            if HAS_CUPY and isinstance(array, cp.ndarray):
                return cp.asnumpy(array)
            return array
    
    def synchronize(self):
        """Synchronize device (useful for timing)"""
        if self.use_cuda:
            cp.cuda.Stream.null.synchronize()
    
    def get_array_module(self, array):
        """Get the array module (numpy or cupy) for given array"""
        if HAS_CUPY:
            return cp.get_array_module(array)
        return np


# Global backend instance(module-level global)
_backend = None


def get_backend(use_cuda=None):
    """Get global backend instance"""
    global _backend  # declare global to modify it
    if _backend is None:
        _backend = Backend(use_cuda=use_cuda)
    return _backend


def set_backend(use_cuda):
    """Set global backend (CPU or CUDA)"""
    global _backend
    _backend = Backend(use_cuda=use_cuda)
    return _backend


def is_cuda_available():
    """Check if CUDA is available"""
    return CUDA_AVAILABLE


def get_device():
    """Get current device"""
    backend = get_backend()
    return backend.device
