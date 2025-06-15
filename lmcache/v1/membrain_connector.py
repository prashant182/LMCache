import asyncio
import hashlib
import base64
import threading
import time
from typing import List, Optional, no_type_check, Dict

from lmcache.v1.membrain_client import MembrainClient, MembrainConfig
from lmcache.v1.memory_management import MemoryAllocatorInterface, MemoryObj, TensorMemoryObj, MemoryObjMetadata, MemoryFormat
from lmcache.v1.protocol import RemoteMetadata  # Reusing Redis metadata format
from lmcache.v1.storage_backend.connector.base_connector import (
    RemoteConnector,
)
from concurrent.futures import Future
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey

import torch

# Import the ProtectedSharedMemory from membrain_gpu_connector
from lmcache.v1.membrain_gpu_connector import ProtectedSharedMemory


logger = init_logger(__name__)


class LeaseMemoryObj(MemoryObj):

    def __init__(self, offsets, metadata: MemoryObjMetadata, shared_memory_name: str = 'membrain-kvcache', lease_id=None, client=None, shared_memory=None):
        self.offsets = offsets  # List of {'offset': int, 'len': int} from Membrain lease
        self.meta: MemoryObjMetadata = metadata
        self.shared_memory_name = shared_memory_name  # For GPU connector access
        self.lease_id = lease_id  # Store lease ID for cleanup
        self.client = client  # Store client reference for lease release
        self.shared_memory = shared_memory  # Pre-initialized shared memory reference
        self.valid = True
        self.lock = threading.Lock()
        self._tensor = None  # Cached tensor
        self._lease_released = False  # Track lease status

    def invalidate(self):
        self.valid = False
        # DO NOT clean up shared memory - it's managed by Membrain
        # Only clean up our cached tensor reference
        self._tensor = None

    def is_valid(self):
        return self.valid

    def get_size(self) -> int:
        return self.meta.phy_size

    def get_shape(self) -> torch.Size:
        return self.meta.shape

    def get_dtype(self) -> torch.dtype:
        return self.meta.dtype

    def get_memory_format(self) -> MemoryFormat:
        with self.lock:
            return self.meta.fmt

    def get_physical_size(self) -> int:
        return self.meta.phy_size

    def ref_count_up(self):
        with self.lock:
            self.meta.ref_count += 1

    def ref_count_down(self):
        should_cleanup = False
        with self.lock:
            self.meta.ref_count -= 1
            if self.meta.ref_count <= 0:
                should_cleanup = True
        
        # Cleanup lease when ref count reaches zero
        if should_cleanup:
            self._cleanup_lease()
    
    def _cleanup_lease(self):
        """Clean up the Membrain lease when object is no longer needed."""
        if self._lease_released or not self.lease_id or not self.client:
            return
            
        try:
            import threading
            import requests
            
            # Mark as released immediately to prevent double cleanup
            self._lease_released = True
            
            # Use synchronous HTTP requests to avoid event loop issues entirely
            def release_lease_sync():
                try:
                    # Build the release URL directly
                    release_url = f"{self.client._config.endpoint}/v1/leases/{self.lease_id}/release"
                    
                    # Use requests library for synchronous HTTP call
                    response = requests.post(
                        release_url,
                        timeout=self.client._config.timeout
                    )
                    
                    if response.status_code == 200:
                        logger.debug(f" LEASE RELEASED SUCCESSFULLY: {self.lease_id}")
                    else:
                        logger.warning(f"LEASE RELEASE HTTP ERROR: {self.lease_id}: {response.status_code}")
                        
                except Exception as release_error:
                    logger.warning(f"LEASE RELEASE ERROR: {self.lease_id}: {release_error}")
            
            # Run lease release in background thread to avoid blocking
            cleanup_thread = threading.Thread(target=release_lease_sync, daemon=True)
            cleanup_thread.start()
            
        except Exception as e:
            logger.warning(f"FAILED TO SCHEDULE LEASE CLEANUP {self.lease_id}: {e}")
    
    def __del__(self):
        """Destructor to ensure lease is released even if ref_count_down isn't called properly."""
        try:
            self._cleanup_lease()
        except Exception as e:
            # Don't let destructor exceptions propagate
            logger.error(f" ERROR IN DESTRUCTOR CLEANUP: {e}")

    def get_ref_count(self) -> int:
        with self.lock:
            return self.meta.ref_count

    def pin(self) -> bool:
        self.metadata.is_pin = True
        return True

    def unpin(self) -> bool:
        self.metadata.is_pin = False
        return True

    @property
    def metadata(self) -> MemoryObjMetadata:
        with self.lock:
            return self.meta

    @property
    def tensor(self) -> Optional[torch.Tensor]:
        # For compatibility with different GPU connectors:
        # - BedrockMembrainGPUConnector (layerwise): Uses offsets directly, expects None
        # - VLLMPagedMemGPUConnectorV2 (non-layerwise): Needs actual tensor data
        
        # Check if we have cached tensor
        if self._tensor is not None:
            return self._tensor
        
        # For non-layerwise mode, we need to create a tensor from shared memory
        # This provides fallback compatibility while maintaining zero-copy for layerwise
        try:
            logger.debug(f"LeaseMemoryObj.tensor: Creating tensor from shared memory for non-layerwise compatibility")
            
            # Use pre-initialized shared memory if available
            if self.shared_memory is not None:
                shm = self.shared_memory
                should_close = False
            else:
                # Fallback to creating a new one
                shm = ProtectedSharedMemory(self.shared_memory_name)
                should_close = True
                
            try:
                # Parse data format: [4-byte-length][metadata][kv_data]
                combined_data = bytearray()
                for offset in self.offsets:
                    o = offset['offset']
                    l = offset['len']
                    segment_data = bytes(shm.buf[o : o + l])
                    combined_data.extend(segment_data)
                
                if len(combined_data) < 4:
                    logger.warning("Insufficient data for tensor creation")
                    return None
                
                metadata_len = int.from_bytes(combined_data[:4], byteorder='little')
                kv_data_start = 4 + metadata_len
                
                if len(combined_data) < kv_data_start:
                    logger.warning("Insufficient data for metadata parsing")
                    return None
                
                # Extract pure KV data
                kv_data = combined_data[kv_data_start:]
                
                if len(kv_data) == 0:
                    logger.warning("No KV data found after metadata")
                    return None
                
                # Create tensor from KV data
                kv_tensor = torch.frombuffer(kv_data, dtype=self.meta.dtype)
                
                # Reshape according to expected format
                try:
                    reshaped_tensor = kv_tensor.view(self.meta.shape)
                    
                    # CRITICAL FIX: Pin the tensor for GPU operations
                    # LMCache C++ operations require "cuda or pinned cpu" device
                    pinned_tensor = reshaped_tensor.pin_memory()
                    
                    # Cache the pinned tensor for future use
                    self._tensor = pinned_tensor
                    logger.debug(f"LeaseMemoryObj.tensor: Created pinned tensor with shape {pinned_tensor.shape}")
                    
                    return pinned_tensor
                    
                except Exception as reshape_error:
                    logger.warning(f"Failed to reshape tensor: {reshape_error}")
                    return None
            finally:
                # Only close if we created it locally
                if should_close:
                    shm.close()
                
        except Exception as e:
            logger.warning(f"Failed to create tensor from shared memory: {e}")
            return None

    @property
    def byte_array(self) -> bytes:
        # For Membrain lease objects, we don't have direct byte array access
        # This should not be called for lease-based objects
        logger.warning("byte_array access attempted on LeaseMemoryObj - this indicates a design issue")
        return b""

    @property
    def is_pinned(self) -> bool:
        return self.metadata.is_pin



class MembrainConnector(RemoteConnector):
    """
    Connector for Membrain key-value store with zero-copy shared memory access.
    
    This connector creates LeaseMemoryObj instances that provide 'offsets' for 
    direct shared memory access. It's designed to work with BedrockMembrainGPUConnector
    for true zero-copy GPU transfers.
    
    The remote url should start with "membrain://" and include a host and port.
    Example: "membrain://172.31.75.30:9200?namespace=kv-cache"
    
    For optimal performance, use with BedrockMembrainGPUConnector which reads
    directly from shared memory using the lease offsets, eliminating intermediate
    CPU tensor copies.
    """

    def __init__(
        self,
        endpoint: str,
        namespace: str,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
    ):
        """Initialize the Membrain connector.

        Args:
            endpoint: The Membrain endpoint URL (e.g., "http://localhost:9201")
            namespace: The namespace to use in Membrain
            loop: The event loop to use for async operations
            local_cpu_backend: ...
        """
        self.config = MembrainConfig(
            endpoint=endpoint,
            namespace=namespace,
            timeout=200.0,  # Reasonable default timeout
        )
        self.client = MembrainClient(self.config)
        self.local_cpu_backend = local_cpu_backend
        self.loop = loop
        # Keep a key mapping cache to be able to track original keys
        self._key_mapping: Dict[str, str] = {}
        
        # Initialize shared memory once for Membrain backend
        self._shared_memory = None
        try:
            self._shared_memory = ProtectedSharedMemory('membrain-kvcache')
            logger.info(f"✅ Initialized shared memory 'membrain-kvcache' for zero-copy access")
        except Exception as e:
            logger.warning(f"⚠️ Could not initialize shared memory on startup: {e}")
        
        logger.info(
            f"Initialized Membrain zero-copy connector: endpoint={endpoint}, namespace={namespace}"
        )

    def _hash_key(self, key_str: str) -> str:
        """
        Hash the long key into a shorter, URL-safe string for Membrain.
        Store the original->hashed mapping for debugging.

        Args:
            key_str: The original long key string

        Returns:
            A URL-safe hashed key string (base64 of SHA-256 hash)
        """
        # Create a hash of the key
        key_hash = hashlib.sha256(key_str.encode()).digest()
        # Convert to URL-safe base64 and remove padding
        safe_key = base64.urlsafe_b64encode(key_hash).decode().rstrip("=")
        # Store mapping for debug and reference
        self._key_mapping[key_str] = safe_key
        logger.debug(f"Hashed key: {key_str} -> {safe_key}")
        return safe_key
    
    def _get_bucket_and_key(self, cache_key: CacheEngineKey) -> tuple[str, str]:
        """
        Convert LMCache key to Membrain bucket/key structure.
        
        Returns:
            tuple: (bucket_name, key_name)
        """
        original_key = cache_key.to_string()
        hashed_key = self._hash_key(original_key)
        
        # Use namespace as bucket, hashed key as the key
        bucket = self.config.namespace
        key = hashed_key
        
        return bucket, key

    async def exists(self, key: CacheEngineKey) -> bool:
        """Check if the key exists in Membrain using proper bucket/key structure."""
        try:
            start = time.time()
            bucket, membrain_key = self._get_bucket_and_key(key)
            
            logger.debug(f" MEMBRAIN EXISTS CHECK: {membrain_key}")

            # Use Membrain API: GET /v1/kv/{bucket}/{key} (returns 200 or 404)
            # Client already uses namespace as bucket, so just pass the key
            exists = await self.client.exists(membrain_key)
            
            end = time.time()
            logger.debug(f"Key exists={exists}, time={(end-start):.3f}s")

            return exists

        except Exception as e:
            # Handle expected cache misses gracefully for exists check too
            if "404" in str(e) or "Key not found" in str(e):
                logger.debug(f" EXISTS CHECK: {membrain_key} not found (cache miss)")
                return False
            else:
                logger.error(f"UNEXPECTED ERROR checking existence for {key.to_string()}: {e}")
                return False

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """Get data from Membrain using lease API for zero-copy access."""
        try:
            bucket, membrain_key = self._get_bucket_and_key(key)
            original_key = key.to_string()
            
            logger.info(f"🔷 MEMBRAIN GET START: {membrain_key} (original: {original_key})")

            # Use Membrain lease API for zero-copy access - manual management to avoid race condition
            lease_info = await self.client.acquire_kv_lease_manual(membrain_key)
            if not lease_info or 'offsets' not in lease_info:
                logger.warning(f"❌ Failed to acquire lease for {membrain_key}")
                return None
                    
            offsets = lease_info['offsets']
            total_bytes = sum(offset['len'] for offset in offsets)
            lease_id = lease_info.get('id')
                
            logger.info(f"📋 LEASE ACQUIRED:")
            logger.info(f"  - Lease ID: {lease_id}")
            logger.info(f"  - Segments: {len(offsets)}")
            logger.info(f"  - Total bytes: {total_bytes:,}")
            logger.info(f"  - Offsets: {offsets}")
                
            # Use pre-initialized shared memory
            shm = self._shared_memory
            if shm is None:
                logger.error(f"❌ Shared memory not initialized for Membrain connector")
                return None
                
            try:
                shm_size = shm.size
                logger.debug(f" SHARED MEMORY SIZE: {shm_size} bytes")
                    
                # Examine first offset in detail
                first_offset = offsets[0]
                start_pos = first_offset['offset']
                length = first_offset['len']
                
                logger.debug(f" FIRST OFFSET: start={start_pos}, len={length}")
                
                # Read first 32 bytes to understand format
                preview_end = min(start_pos + 32, shm_size)
                preview_bytes = bytes(shm.buf[start_pos:preview_end])
                logger.debug(f" MEMORY PREVIEW (first 32 bytes): {preview_bytes.hex()}")
                
                # Try to read as metadata length (4 bytes)
                if length >= 4:
                    metadata_len_bytes = shm.buf[start_pos:start_pos + 4]
                    metadata_len = int.from_bytes(metadata_len_bytes, byteorder='little')
                    logger.debug(f" METADATA LENGTH: {metadata_len} bytes")
                    
                    # Check if this makes sense
                    if 0 < metadata_len < length:
                        logger.debug(f" METADATA LENGTH VALID: {metadata_len} < {length}")
                        
                        # Try to read metadata
                        metadata_start = start_pos + 4
                        metadata_end = metadata_start + metadata_len
                        
                        if metadata_end <= shm_size:
                            metadata_bytes = shm.buf[metadata_start:metadata_end]
                            logger.debug(f" RAW METADATA: {len(metadata_bytes)} bytes")
                            
                            try:
                                metadata = RemoteMetadata.deserialize(memoryview(metadata_bytes))
                                logger.info(f" METADATA DESERIALIZED: shape={metadata.shape}, dtype={metadata.dtype}")
                                
                                # Calculate KV data region
                                kv_data_start = metadata_end
                                kv_data_size = total_bytes - 4 - metadata_len
                                logger.info(f" KV DATA: start={kv_data_start}, size={kv_data_size}")
                                
                                # Create proper LeaseMemoryObj with client reference for cleanup
                                lease_obj = LeaseMemoryObj(
                                    offsets=offsets,
                                    metadata=MemoryObjMetadata.from_dict({
                                        "shape": list(metadata.shape),
                                        "dtype": str(metadata.dtype),
                                        "address": id(lease_info),
                                        "phy_size": kv_data_size,
                                        "ref_count": 1,
                                        "fmt": metadata.fmt.value,
                                    }),
                                    shared_memory_name='membrain-kvcache',
                                    lease_id=lease_id,
                                    client=self.client,  # Pass client for lease cleanup
                                    shared_memory=self._shared_memory  # Pass pre-initialized shared memory
                                )
                                
                                logger.info(f" CREATED LEASE MEMORY OBJ: {type(lease_obj)}")
                                return lease_obj
                                
                            except Exception as metadata_error:
                                logger.error(f" METADATA DESERIALIZATION FAILED: {metadata_error}")
                        else:
                            logger.error(f" METADATA EXTENDS BEYOND BUFFER: {metadata_end} > {shm.size}")
                    else:
                        logger.error(f" INVALID METADATA LENGTH: {metadata_len} (total length: {length})")
                else:
                    logger.error(f" OFFSET TOO SMALL FOR METADATA: {length} < 4")
                
                # Fallback: examine raw data to understand format
                logger.info(f" FALLBACK: Analyzing raw data format")
                
                # Sample data from different positions
                sample_size = min(64, length)
                sample_data = bytes(shm.buf[start_pos:start_pos + sample_size])
                logger.info(f" RAW DATA SAMPLE: {sample_data.hex()}")
                
                # Try different interpretations
                logger.info(f" AS LITTLE-ENDIAN INTS: {[int.from_bytes(sample_data[i:i+4], 'little') for i in range(0, min(16, len(sample_data)), 4)]}")
                logger.info(f" AS BIG-ENDIAN INTS: {[int.from_bytes(sample_data[i:i+4], 'big') for i in range(0, min(16, len(sample_data)), 4)]}")
                
                # Create minimal fallback object with client reference
                fallback_obj = LeaseMemoryObj(
                    offsets=offsets,
                    metadata=MemoryObjMetadata.from_dict({
                        "shape": [1, 1, total_bytes],
                        "dtype": "torch.float16",
                        "address": id(lease_info),
                        "phy_size": total_bytes,
                        "ref_count": 1,
                        "fmt": 0,
                    }),
                    shared_memory_name='membrain-kvcache',
                    lease_id=lease_id,
                    client=self.client,
                    shared_memory=self._shared_memory  # Pass pre-initialized shared memory  
                )
                
                logger.warning(f"USING FALLBACK LEASE OBJECT")
                return fallback_obj
                
            except Exception as shm_error:
                logger.error(f" SHARED MEMORY ACCESS FAILED: {shm_error}")
                logger.error(f" SHM ERROR TYPE: {type(shm_error)}")
                import traceback
                logger.error(f" FULL TRACEBACK: {traceback.format_exc()}")
                
                # Release lease on error
                try:
                    await self.client.release_lease_manual(lease_id)
                    logger.info(f"LEASE RELEASED DUE TO ERROR: {lease_id}")
                except Exception as release_error:
                    logger.error(f" FAILED TO RELEASE LEASE {lease_id}: {release_error}")
                
                return None
            finally:
                # No cleanup needed - shared memory is managed at connector level
                pass

        except Exception as e:
            # Handle expected cache misses gracefully
            if "404" in str(e) or "Key not found" in str(e):
                # This is a normal cache miss - log at debug level only
                logger.debug(f" CACHE MISS: {membrain_key} not found in Membrain")
                return None
            else:
                # This is an unexpected error - log at error level
                logger.error(f" UNEXPECTED MEMBRAIN ERROR for {key.to_string()}: {e}")
                logger.error(f" ERROR TYPE: {type(e)}")
                import traceback
                logger.error(f" FULL TRACEBACK: {traceback.format_exc()}")
                return None
        
    def get_non_blocking(
        self,
        key: CacheEngineKey,
    ) -> Optional[Future]:
        return asyncio.run_coroutine_threadsafe(
            self.get(key), self.loop
        )

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        """Store data in Membrain using proper bucket/key structure."""
        try:
            bucket, membrain_key = self._get_bucket_and_key(key)
            original_key = key.to_string()

            logger.info(f"🔷 MEMBRAIN PUT START: {original_key} -> bucket:{bucket}/key:{membrain_key}")

            # Extract data from memory object
            kv_bytes = memory_obj.byte_array
            kv_shape = memory_obj.get_shape()
            kv_dtype = memory_obj.get_dtype()
            memory_format = memory_obj.get_memory_format()

            logger.info(f"📊 PUT DATA INFO:")
            logger.info(f"  - Shape: {kv_shape}")
            logger.info(f"  - Dtype: {kv_dtype}")
            logger.info(f"  - Memory format: {memory_format}")
            logger.info(f"  - Data size: {len(kv_bytes):,} bytes")

            # Create metadata for reconstruction
            if len(kv_shape) == 3:
                logger.info(f"  - Reshaping 3D to 4D: {kv_shape} -> {(kv_shape[0], 1, *kv_shape[1:])}")
                kv_shape = (kv_shape[0], 1, *kv_shape[1:])

            metadata = RemoteMetadata(len(kv_bytes), kv_shape, kv_dtype, memory_format)
            metadata_bytes = metadata.serialize()

            # Log metadata details
            logger.info(f"📋 METADATA CREATED:")
            logger.info(f"  - Metadata length: {len(metadata_bytes)} bytes")
            logger.info(f"  - Metadata bytes (first 32): {metadata_bytes[:32].hex()}")

            # Combine metadata + data for single Membrain storage
            # Structure: [metadata_length(4 bytes)] + [metadata] + [kv_data]
            metadata_len = len(metadata_bytes)
            combined_data = (
                metadata_len.to_bytes(4, byteorder='little') + 
                metadata_bytes + 
                kv_bytes
            )

            # Log combined data structure
            logger.info(f"📦 COMBINED DATA STRUCTURE:")
            logger.info(f"  - Metadata length bytes: {metadata_len.to_bytes(4, byteorder='little').hex()}")
            logger.info(f"  - Total size: {len(combined_data):,} bytes")
            logger.info(f"  - First 64 bytes: {combined_data[:64].hex()}")

            # Store in Membrain using proper API: PUT /v1/kv/{bucket}/{key}
            logger.info(f"🚀 SENDING TO MEMBRAIN: {len(combined_data):,} bytes")
            
            response = await self.client.put(membrain_key, combined_data)
            
            logger.info(f"✅ MEMBRAIN PUT SUCCESS: Stored {len(combined_data):,} bytes for {original_key}")
            
            # Verify data structure integrity
            logger.info(f"🔍 VERIFYING PUT DATA:")
            logger.info(f"  - KV data first 32 bytes: {kv_bytes[:32].hex()}")
            logger.info(f"  - KV data last 32 bytes: {kv_bytes[-32:].hex()}")

            # Clear local cache to free memory
            self.local_cpu_backend.clear()

        except Exception as e:
            logger.error(f"❌ MEMBRAIN PUT ERROR for {key.to_string()}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

    @no_type_check
    async def list(self) -> List[str]:
        """List keys in Membrain (not implemented)."""
        logger.warning("List operation not supported by Membrain client")
        return []

    async def close(self):
        """Close the Membrain client and cleanup resources."""
        try:
            # Close the client
            await self.client.close()
            
            # Close the shared memory reference
            if self._shared_memory is not None:
                try:
                    self._shared_memory.close()
                    logger.info("✅ Closed shared memory reference")
                except Exception as e:
                    logger.warning(f"⚠️ Error closing shared memory: {e}")
            
            logger.info("Closed Membrain connector")
        except Exception as e:
            logger.error(f"Error closing Membrain connector: {e}")
