"""
Simplified Membrain connector with streamlined LeaseMemoryObj.
Focuses on zero-copy access without unnecessary complexity.
"""

import asyncio
import hashlib
import base64
import logging
import requests
import threading
from typing import List, Optional, Dict, Tuple
from concurrent.futures import Future

import torch

from lmcache.v1.membrain_client import MembrainClient, MembrainConfig
from lmcache.v1.memory_management import (
    MemoryObj, MemoryObjMetadata, MemoryFormat
)
from lmcache.v1.protocol import RemoteMetadata
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.utils import CacheEngineKey
from lmcache.v1.shared_memory_manager import SharedMemoryManager

logger = logging.getLogger(__name__)


class SimplifiedLeaseMemoryObj(MemoryObj):
    """
    Lightweight lease object for zero-copy access.
    
    This simplified version removes unnecessary tensor creation logic
    and focuses on providing the minimal interface needed for zero-copy
    GPU transfers.
    """
    
    def __init__(
        self, 
        offsets: List[Dict[str, int]], 
        metadata: MemoryObjMetadata,
        lease_id: Optional[str] = None,
        client: Optional[MembrainClient] = None
    ):
        """
        Initialize lease memory object.
        
        Args:
            offsets: List of offset dictionaries from Membrain lease
            metadata: Memory object metadata
            lease_id: Lease ID for cleanup
            client: Membrain client for lease release
        """
        self.offsets = offsets
        self._metadata = metadata  # Store as private attribute
        self.lease_id = lease_id
        self._client = client
        self._lease_released = False
    
    def invalidate(self):
        """Mark object as invalid."""
        pass  # No cleanup needed for lease objects
    
    def is_valid(self) -> bool:
        """Check if object is valid."""
        return not self._lease_released
    
    def get_size(self) -> int:
        """Get physical size."""
        return self._metadata.phy_size
    
    def get_shape(self) -> torch.Size:
        """Get tensor shape."""
        return self._metadata.shape
    
    def get_dtype(self) -> torch.dtype:
        """Get tensor dtype."""
        return self._metadata.dtype
    
    def get_memory_format(self) -> MemoryFormat:
        """Get memory format."""
        return self._metadata.fmt
    
    def get_physical_size(self) -> int:
        """Get physical size."""
        return self._metadata.phy_size
    
    def ref_count_up(self):
        """Increment reference count."""
        self._metadata.ref_count += 1
    
    def ref_count_down(self):
        """Decrement reference count and cleanup if needed."""
        self._metadata.ref_count -= 1
        if self._metadata.ref_count <= 0:
            self._release_lease()
    
    def _release_lease(self):
        """Release the Membrain lease asynchronously."""
        if self._lease_released or not self.lease_id or not self._client:
            return
        
        self._lease_released = True
        
        # Async lease release in background thread
        def release_async():
            try:
                release_url = f"{self._client._config.endpoint}/v1/leases/{self.lease_id}/release"
                response = requests.post(release_url, timeout=5.0)
                if response.status_code == 200:
                    logger.debug(f"Released lease: {self.lease_id}")
                else:
                    logger.warning(f"Lease release failed: {self.lease_id}")
            except Exception as e:
                logger.warning(f"Lease release error: {e}")
        
        threading.Thread(target=release_async, daemon=True).start()
    
    def __del__(self):
        """Ensure lease is released on deletion."""
        try:
            self._release_lease()
        except Exception:
            pass
    
    def get_ref_count(self) -> int:
        """Get current reference count."""
        return self._metadata.ref_count
    
    def pin(self) -> bool:
        """Pin memory (not applicable for lease objects)."""
        return True
    
    def unpin(self) -> bool:
        """Unpin memory (not applicable for lease objects)."""
        return True
    
    @property
    def tensor(self) -> None:
        """Return None for zero-copy path."""
        return None
    
    @property
    def byte_array(self) -> bytes:
        """Not applicable for lease objects."""
        return b""
    
    @property
    def is_pinned(self) -> bool:
        """Check if pinned."""
        return self._metadata.is_pin
    
    @property
    def metadata(self) -> MemoryObjMetadata:
        """Get the metadata of the MemoryObj."""
        return self._metadata


class SimplifiedMembrainConnector(RemoteConnector):
    """
    Simplified connector for Membrain with zero-copy shared memory access.
    
    This connector creates lightweight lease objects that provide offsets
    for direct shared memory access, optimized for use with the streamlined
    Membrain GPU connector.
    """
    
    def __init__(
        self,
        endpoint: str,
        namespace: str,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend
    ):
        """
        Initialize the Membrain connector.
        
        Args:
            endpoint: Membrain endpoint URL
            namespace: Membrain namespace
            loop: Event loop for async operations
            local_cpu_backend: Local CPU backend (for compatibility)
        """
        self.config = MembrainConfig(
            endpoint=endpoint,
            namespace=namespace,
            timeout=30.0
        )
        self.client = MembrainClient(self.config)
        self.local_cpu_backend = local_cpu_backend
        self.loop = loop
        self._key_cache: Dict[str, str] = {}
        
        # Initialize shared memory manager once
        self._shm_manager = SharedMemoryManager('membrain-kvcache')
        if self._shm_manager.attach():
            logger.info(f"✅ Membrain connector: Attached to shared memory")
        else:
            logger.warning(f"Membrain connector: Could not attach to shared memory")
    
    def _hash_key(self, key_str: str) -> str:
        """Hash long keys into URL-safe strings."""
        if key_str in self._key_cache:
            return self._key_cache[key_str]
        
        key_hash = hashlib.sha256(key_str.encode()).digest()
        safe_key = base64.urlsafe_b64encode(key_hash).decode().rstrip("=")
        self._key_cache[key_str] = safe_key
        return safe_key
    
    def _get_bucket_and_key(self, cache_key: CacheEngineKey) -> Tuple[str, str]:
        """Convert LMCache key to Membrain bucket/key."""
        original_key = cache_key.to_string()
        hashed_key = self._hash_key(original_key)
        return self.config.namespace, hashed_key
    
    async def exists(self, key: CacheEngineKey) -> bool:
        """Check if key exists in Membrain."""
        try:
            _, membrain_key = self._get_bucket_and_key(key)
            return await self.client.exists(membrain_key)
        except Exception as e:
            if "404" in str(e) or "not found" in str(e).lower():
                return False
            logger.error(f"Error checking existence: {e}")
            return False
    
    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """Get data from Membrain using lease API."""
        try:
            _, membrain_key = self._get_bucket_and_key(key)
            
            # Acquire lease for zero-copy access
            lease_info = await self.client.acquire_kv_lease_manual(membrain_key)
            if not lease_info or 'offsets' not in lease_info:
                return None
            
            offsets = lease_info['offsets']
            lease_id = lease_info.get('id')
            
            # Parse metadata to create proper MemoryObjMetadata
            shm_buffer = self._shm_manager.get_buffer()
            if not shm_buffer:
                logger.error("Failed to access shared memory")
                return None
            
            # Extract metadata from first segment
            first_offset = offsets[0]
            metadata = self._extract_metadata(
                first_offset, shm_buffer, sum(o['len'] for o in offsets)
            )
            
            if not metadata:
                # Release lease on failure
                try:
                    await self.client.release_lease_manual(lease_id)
                except Exception:
                    pass
                return None
            
            # Create lightweight lease object
            return SimplifiedLeaseMemoryObj(
                offsets=offsets,
                metadata=metadata,
                lease_id=lease_id,
                client=self.client
            )
            
        except Exception as e:
            if "404" in str(e) or "not found" in str(e).lower():
                logger.debug(f"Cache miss: {membrain_key}")
                return None
            logger.error(f"Membrain get error: {e}")
            return None
    
    def _extract_metadata(
        self, 
        first_offset: Dict[str, int],
        shm_buffer: memoryview,
        total_size: int
    ) -> Optional[MemoryObjMetadata]:
        """Extract metadata from shared memory."""
        try:
            offset = first_offset['offset']
            length = first_offset['len']
            
            if length < 4:
                return None
            
            # Read metadata length
            metadata_len = int.from_bytes(
                shm_buffer[offset:offset + 4], 
                byteorder='little'
            )
            
            if metadata_len <= 0 or metadata_len > length - 4:
                return None
            
            # Read metadata
            metadata_bytes = shm_buffer[offset + 4:offset + 4 + metadata_len]
            remote_metadata = RemoteMetadata.deserialize(metadata_bytes)
            
            # Calculate KV data size
            kv_data_size = total_size - 4 - metadata_len
            
            # Create MemoryObjMetadata
            return MemoryObjMetadata.from_dict({
                "shape": list(remote_metadata.shape),
                "dtype": str(remote_metadata.dtype),
                "address": 0,  # Not used for lease objects
                "phy_size": kv_data_size,
                "ref_count": 1,
                "fmt": remote_metadata.fmt.value,
            })
            
        except Exception as e:
            logger.error(f"Failed to extract metadata: {e}")
            return None
    
    def get_non_blocking(self, key: CacheEngineKey) -> Optional[Future]:
        """Non-blocking get operation."""
        return asyncio.run_coroutine_threadsafe(self.get(key), self.loop)
    
    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        """Store data in Membrain."""
        try:
            _, membrain_key = self._get_bucket_and_key(key)
            
            # Extract data from memory object
            kv_bytes = memory_obj.byte_array
            kv_shape = memory_obj.get_shape()
            kv_dtype = memory_obj.get_dtype()
            memory_format = memory_obj.get_memory_format()
            
            # Handle 3D to 4D conversion if needed
            if len(kv_shape) == 3:
                kv_shape = (kv_shape[0], 1, *kv_shape[1:])
            
            # Create metadata
            metadata = RemoteMetadata(
                len(kv_bytes), kv_shape, kv_dtype, memory_format
            )
            metadata_bytes = metadata.serialize()
            
            # Combine metadata + data
            combined_data = (
                len(metadata_bytes).to_bytes(4, byteorder='little') +
                metadata_bytes +
                kv_bytes
            )
            
            # Store in Membrain
            await self.client.put(membrain_key, combined_data)
            
            logger.debug(f"Stored {len(combined_data)} bytes for key {membrain_key}")
            
            # Clear local cache
            self.local_cpu_backend.clear()
            
        except Exception as e:
            logger.error(f"Membrain put error: {e}")
    
    async def list(self) -> List[str]:
        """List operation not supported."""
        return []
    
    async def close(self):
        """Close connector and cleanup resources."""
        try:
            await self.client.close()
            if self._shm_manager:
                self._shm_manager.close()
            logger.info("Closed Membrain connector")
        except Exception as e:
            logger.error(f"Error closing connector: {e}")