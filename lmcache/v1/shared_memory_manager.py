"""
Centralized shared memory management for Membrain zero-copy backend.
Handles all SharedMemory lifecycle and protection logic in one place.
"""

import logging
from typing import Optional
from multiprocessing.shared_memory import SharedMemory
from multiprocessing import resource_tracker

logger = logging.getLogger(__name__)


class SharedMemoryManager:
    """
    Centralized shared memory management with protection against Python cleanup bugs.
    
    This class provides a single point of management for shared memory access,
    preventing the Python SharedMemory automatic cleanup bug that can corrupt
    memory when exceptions occur.
    """
    
    def __init__(self, name: str):
        """
        Initialize the shared memory manager.
        
        Args:
            name: The name of the shared memory segment to attach to
        """
        self._name = name
        self._shm: Optional[SharedMemory] = None
        self._attached = False
    
    def attach(self) -> bool:
        """
        Attach to existing shared memory with full protection against cleanup.
        
        Returns:
            True if successfully attached, False otherwise
        """
        if self._attached and self._shm is not None:
            return True
            
        try:
            # Attach to existing shared memory - never create or unlink
            self._shm = SharedMemory(name=self._name, create=False)
            self._attached = True
            
            # Apply all protection mechanisms against automatic cleanup
            self._protect_from_cleanup()
            
            logger.debug(f"Successfully attached to SharedMemory '{self._name}'")
            return True
            
        except FileNotFoundError:
            logger.warning(f"SharedMemory '{self._name}' not found - may not be initialized yet")
            self._attached = False
            return False
        except Exception as e:
            logger.error(f"Failed to attach to SharedMemory '{self._name}': {e}")
            self._attached = False
            return False
    
    def _protect_from_cleanup(self):
        """Apply multiple protection layers against automatic memory cleanup."""
        if not self._shm:
            return
            
        # 1. Replace __del__ to prevent automatic cleanup
        def safe_del(self):
            pass
        self._shm.__del__ = safe_del
        
        # 2. Replace close() to prevent unlinking
        original_close = self._shm.close
        def safe_close():
            try:
                if hasattr(self._shm, '_mmap') and self._shm._mmap is not None:
                    self._shm._mmap.close()
            except Exception:
                pass
        self._shm.close = safe_close
        
        # 3. Replace unlink() to prevent accidental unlinking
        def no_unlink():
            logger.warning(f"Attempted to unlink shared memory '{self._name}' - BLOCKED")
        self._shm.unlink = no_unlink
        
        # 4. Unregister from resource tracker to prevent cleanup
        try:
            resource_tracker.unregister(self._name, 'shared_memory')
        except KeyError:
            pass
        except Exception as e:
            logger.debug(f"Resource tracker unregister failed (non-critical): {e}")
        
        # 5. Clear the _name attribute as additional protection
        self._shm._name = None
    
    def get_buffer(self) -> Optional[memoryview]:
        """
        Get memory buffer, auto-attaching if needed.
        
        Returns:
            Memory buffer view or None if not available
        """
        if not self._attached or self._shm is None:
            if not self.attach():
                return None
        
        if self._shm and hasattr(self._shm, 'buf'):
            return self._shm.buf
        return None
    
    @property
    def size(self) -> int:
        """Get the size of the shared memory segment."""
        if not self._attached or self._shm is None:
            self.attach()
        
        if self._shm and hasattr(self._shm, 'size'):
            return self._shm.size
        return 0
    
    def close(self):
        """Safe close without unlinking the shared memory."""
        if self._shm is not None:
            try:
                # Use our safe_close method
                if hasattr(self._shm, 'close'):
                    self._shm.close()
                logger.debug(f"Closed SharedMemory '{self._name}' reference safely")
            except Exception as e:
                logger.debug(f"Error during close (expected): {e}")
            finally:
                self._shm = None
                self._attached = False
    
    def __del__(self):
        """Ensure cleanup on deletion."""
        self.close()