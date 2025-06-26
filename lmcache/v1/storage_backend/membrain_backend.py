# Copyright 2024-2025 LMCache Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Standard
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import List, Optional, Tuple
import asyncio
import json
import numpy as np
import threading

# Third Party
import aiohttp
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey, _lmcache_nvtx_annotate
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryAllocatorInterface, MemoryObj, MemoryFormat
from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface

logger = init_logger(__name__)


class DTypeManager:
    """Centralized dtype management for MembrainBackend serialization.
    
    Handles torch.dtype <-> numpy.dtype conversions and serialization compatibility.
    Single source of truth for all dtype-related operations.
    """
    
    # Direct serializable dtypes (no conversion needed for numpy compatibility)
    DIRECT_SERIALIZABLE = {
        torch.float32: np.float32,
        torch.float16: np.float16,
        torch.int32: np.int32,
        torch.int64: np.int64,
        torch.int16: np.int16,
        torch.int8: np.int8,
        torch.uint8: np.uint8,
        torch.bool: np.bool_,
    }
    
    # Dtypes that can be stored as raw bytes without size expansion
    RAW_BYTES_DTYPES = {
        torch.bfloat16: np.uint16,  # Store bf16 as uint16 (same 16-bit size)
    }
    
    # Dtypes that require conversion for numpy/serialization compatibility
    CONVERSION_REQUIRED = {
        torch.float8_e4m3fn: torch.float32, # float8 -> float32
        torch.float8_e5m2: torch.float32,  # float8 -> float32
        torch.complex64: torch.float32,    # complex -> float32 (flatten to real)
        torch.complex128: torch.float32,   # complex -> float32
    }
    
    @classmethod
    def get_serialization_info(cls, original_dtype: torch.dtype) -> Tuple[torch.dtype, np.dtype]:
        """Get serialization dtype and corresponding numpy dtype for storage.
        
        Args:
            original_dtype: The original tensor dtype
            
        Returns:
            Tuple of (serialized_torch_dtype, numpy_dtype_for_storage)
        """
        if original_dtype in cls.DIRECT_SERIALIZABLE:
            # Can serialize directly without conversion
            numpy_dtype = cls.DIRECT_SERIALIZABLE[original_dtype]
            return original_dtype, numpy_dtype
        elif original_dtype in cls.RAW_BYTES_DTYPES:
            # Store as raw bytes using equivalent-sized numpy dtype
            numpy_dtype = cls.RAW_BYTES_DTYPES[original_dtype]
            return original_dtype, numpy_dtype  # Keep original dtype for metadata
        elif original_dtype in cls.CONVERSION_REQUIRED:
            # Requires conversion for serialization
            serialized_dtype = cls.CONVERSION_REQUIRED[original_dtype]
            numpy_dtype = cls.DIRECT_SERIALIZABLE[serialized_dtype]
            return serialized_dtype, numpy_dtype
        else:
            # Unknown dtype - fallback to float32 with warning
            logger.warning(f"Unknown dtype {original_dtype}, falling back to float32 for serialization")
            return torch.float32, np.float32
    
    @classmethod
    def get_numpy_dtype_from_torch(cls, torch_dtype: torch.dtype) -> np.dtype:
        """Get numpy dtype from torch dtype (for deserialization)."""
        _, numpy_dtype = cls.get_serialization_info(torch_dtype)
        return numpy_dtype
    
    @classmethod
    def requires_conversion(cls, dtype: torch.dtype) -> bool:
        """Check if dtype requires conversion for serialization."""
        return dtype in cls.CONVERSION_REQUIRED
    
    @classmethod
    def requires_raw_bytes_handling(cls, dtype: torch.dtype) -> bool:
        """Check if dtype requires special raw bytes handling."""
        return dtype in cls.RAW_BYTES_DTYPES
    
    @classmethod
    def apply_serialization_conversion(cls, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.dtype]:
        """Apply necessary conversions for serialization.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Tuple of (converted_tensor, serialized_dtype)
        """
        original_dtype = tensor.dtype
        serialized_dtype, _ = cls.get_serialization_info(original_dtype)
        
        if original_dtype in cls.RAW_BYTES_DTYPES:
            # Handle raw bytes dtypes (e.g., bfloat16) by reinterpreting as equivalent uint type
            if original_dtype == torch.bfloat16:
                # Reinterpret bfloat16 data as uint16 - same size, no data expansion
                converted_tensor = tensor.view(torch.uint16)
                logger.debug(f"Reinterpreting {original_dtype} as uint16 for raw bytes serialization")
                return converted_tensor, original_dtype  # Keep original dtype for metadata
            else:
                # Future raw bytes dtypes can be added here
                logger.warning(f"Unhandled raw bytes dtype {original_dtype}, falling back to conversion")
                converted_tensor = tensor.to(torch.float32)
                return converted_tensor, torch.float32
        elif serialized_dtype != original_dtype:
            # Conversion required
            if original_dtype in [torch.complex64, torch.complex128]:
                # Special handling for complex numbers - could take real part or flatten
                logger.debug(f"Converting complex dtype {original_dtype} to {serialized_dtype}")
                converted_tensor = tensor.real.to(serialized_dtype)
            else:
                # Standard conversion
                converted_tensor = tensor.to(serialized_dtype)
            return converted_tensor, serialized_dtype
        else:
            # No conversion needed
            return tensor, original_dtype


@dataclass
class LeaseInfo:
    """Information about a lease obtained from Membrain daemon."""
    lease_id: str
    offsets: List[Tuple[int, int]]  # (offset, length) pairs
    total_size: int


class MembrainBackend(StorageBackendInterface):
    """
    A storage backend that uses Membrain KV cache daemon for layerwise caching.
    
    This backend is designed for layerwise mode operations and provides:
    - Direct shared memory access via leases (no local_cpu_backend buffer)
    - HTTP API integration with Membrain daemon
    - Efficient batch operations for layer-by-layer processing
    - Memory-mapped file access for zero-copy operations
    
    Configuration requires:
    - membrain_url: URL of the Membrain daemon (e.g., "http://localhost:9200")
    - shared_memory_name: Optional name for shared memory segment
    - bucket_name: Bucket name for priority/organization (default: "lmcache")
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        loop: asyncio.AbstractEventLoop,
        memory_allocator: MemoryAllocatorInterface,
        dst_device: str = "cuda",
    ):
        super().__init__(dst_device)
        
        self.config = config
        self.loop = loop
        self.memory_allocator = memory_allocator

        # Membrain configuration
        self.membrain_url = getattr(config, 'membrain_url', 'http://localhost:9200')
        self.shared_memory_name = getattr(config, 'shared_memory_name', None)
        self.bucket_name = getattr(config, 'membrain_bucket', 'lmcache')
        self.timeout_ms = getattr(config, 'membrain_timeout_ms', 5000)

        # Performance optimizations for scale
        self.max_connections = getattr(config, 'membrain_max_connections', 256)
        self.max_connections_per_host = getattr(config, 'membrain_max_connections_per_host', 128)
        self.serialization_threads = getattr(config, 'membrain_serialization_threads', 16)

        # HTTP connection pool for high-scale performance
        self.http_session: Optional[aiohttp.ClientSession] = None
        self.session_lock = asyncio.Lock()
        
        # Thread pool for CPU-bound serialization operations
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.serialization_threads,
            thread_name_prefix="membrain-serialize"
        )

        # Put task tracking - required by interface
        self.put_lock = threading.Lock()
        self.put_tasks: set[CacheEngineKey] = set()

        # Shared memory mapping (lazy initialization)
        self.shared_memory_obj: Optional[shared_memory.SharedMemory] = None
        self.shared_memory_map: Optional[memoryview] = None
        self.shared_memory_lock = threading.Lock()

        logger.info(
            f"MembrainBackend initialized with URL: {self.membrain_url}, "
            f"bucket: {self.bucket_name}, shared_memory: {self.shared_memory_name}, "
            f"max_connections: {self.max_connections}, max_connections_per_host: {self.max_connections_per_host}, "
            f"serialization_threads: {self.serialization_threads}"
        )

    def __str__(self):
        return self.__class__.__name__

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        """Check if key exists in Membrain cache."""
        try:
            key_str = self._key_to_string(key)
            url = f"{self.membrain_url}/v1/kv/{self.bucket_name}/{key_str}/locations"

            # Simplified sync check - no local caching complexity
            result = asyncio.run_coroutine_threadsafe(
                self._http_request('GET', url, timeout=2.0), self.loop
            ).result()

            return result is not None and result.get('status') == 200
        except Exception as e:
            logger.debug(f"Failed to check key existence: {e}")
            return False

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        """Check if key is currently being stored."""
        with self.put_lock:
            return key in self.put_tasks

    @_lmcache_nvtx_annotate
    def batched_submit_put_task(
        self, keys: List[CacheEngineKey], memory_objs: List[MemoryObj]
    ) -> Optional[List[Future]]:
        """Submit batch of PUT tasks to Membrain."""
        for key, memory_obj in zip(keys, memory_objs, strict=False):
            self.submit_put_task(key, memory_obj)
        return None

    @_lmcache_nvtx_annotate
    def submit_put_task(
        self, key: CacheEngineKey, memory_obj: MemoryObj
    ) -> Optional[Future]:
        """Submit a single PUT task to Membrain."""
        memory_obj.ref_count_up()

        with self.put_lock:
            self.put_tasks.add(key)

        self.loop.call_soon_threadsafe(
            asyncio.create_task, self._async_put(key, memory_obj)
        )
        return None

    async def _ensure_http_session(self) -> aiohttp.ClientSession:
        """Ensure HTTP session with connection pooling is initialized."""
        if self.http_session is None:
            async with self.session_lock:
                if self.http_session is None:  # Double-check locking
                    connector = aiohttp.TCPConnector(
                        limit=self.max_connections,                    # Total connection pool size
                        limit_per_host=self.max_connections_per_host, # Per-host connection limit
                        ttl_dns_cache=300,                           # DNS cache TTL (5 min)
                        use_dns_cache=True,                          # Enable DNS caching
                        keepalive_timeout=30,                        # Keep connections alive
                        enable_cleanup_closed=True,                  # Clean up closed connections
                    )

                    timeout = aiohttp.ClientTimeout(
                        total=30,        # Total timeout for request
                        connect=5,       # Connection timeout
                        sock_read=10,    # Socket read timeout
                    )

                    self.http_session = aiohttp.ClientSession(
                        connector=connector,
                        timeout=timeout,
                        headers={'User-Agent': 'LMCache-MembrainBackend/1.0'}
                    )
                    logger.info(f"Created HTTP session with {self.max_connections} max connections")

        return self.http_session

    async def _http_request(self, method: str, url: str, data=None, params=None, timeout=5.0):
        """Optimized HTTP request with connection pooling."""
        try:
            session = await self._ensure_http_session()
            request_timeout = aiohttp.ClientTimeout(total=timeout)

            async with session.request(
                method, url, data=data, params=params, timeout=request_timeout
            ) as response:
                result = {
                    'status': response.status,
                    'data': await response.read() if method in ['PUT', 'POST'] else None,
                    'json': await response.json() if response.content_type == 'application/json' else None
                }
                return result
        except asyncio.TimeoutError:
            logger.warning(f"HTTP {method} request timeout for {url}")
            return None
        except aiohttp.ClientError as e:
            logger.error(f"HTTP {method} client error for {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"HTTP {method} request failed for {url}: {e}")
            return None

    @_lmcache_nvtx_annotate
    async def _async_put(self, key: CacheEngineKey, memory_obj: MemoryObj) -> None:
        """Optimized async PUT operation with thread pool for serialization and early memory release."""
        serialization_start = None
        http_start = None
        memory_released = False

        try:
            key_str = self._key_to_string(key)
            url = f"{self.membrain_url}/v1/kv/{self.bucket_name}/{key_str}"

            # OPTIMIZATION 1: Serialize tensor on thread pool (CPU-bound operation)
            loop = asyncio.get_running_loop()
            serialization_start = loop.time()
            data = await loop.run_in_executor(
                self.thread_pool,
                self._memory_obj_to_bytes,
                memory_obj
            )
            serialization_time = loop.time() - serialization_start

            # OPTIMIZATION 2: Early memory release - tensor copied to bytes, release GPU memory
            memory_obj.ref_count_down()
            memory_released = True

            # HTTP request on event loop (I/O-bound operation)
            http_start = loop.time()
            result = await self._http_request('PUT', url, data=data, timeout=self.timeout_ms/1000.0)
            http_time = loop.time() - http_start

            if result and result['status'] == 200:
                logger.debug(
                    f"Successfully stored key {key}: {len(data)} bytes, "
                    f"serialize: {serialization_time*1000:.1f}ms, http: {http_time*1000:.1f}ms"
                )
            else:
                status = result['status'] if result else 'TIMEOUT'
                logger.error(f"Failed to store key {key}: HTTP {status}")
        except Exception as e:
            logger.exception(f"Exception during PUT for key {key}: {e}")
            # Ensure memory is released even on error
            if not memory_released:
                try:
                    memory_obj.ref_count_down()
                except:
                    pass  # May have already been released or failed for other reasons
        finally:
            # Always cleanup task tracking
            with self.put_lock:
                self.put_tasks.discard(key)

    def submit_prefetch_task(self, key: CacheEngineKey) -> Optional[Future]:
        """Submit prefetch task - unified with other GET operations."""
        return asyncio.run_coroutine_threadsafe(self._get_memory_obj(key), self.loop)

    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """Blocking GET operation from Membrain."""
        try:
            return asyncio.run_coroutine_threadsafe(self._get_memory_obj(key), self.loop).result()
        except Exception as e:
            logger.error(f"GET operation exception for key {key}: {e}")
            return None

    def get_non_blocking(self, key: CacheEngineKey) -> Optional[Future]:
        """Non-blocking GET operation."""
        return asyncio.run_coroutine_threadsafe(self._get_memory_obj(key), self.loop)

    async def _get_memory_obj(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """Unified GET method: lease → read → reconstruct → release."""
        # Step 1: Acquire lease
        lease_info = await self._acquire_lease(key)
        if lease_info is None:
            return None

        try:
            # Step 2: Read and reconstruct tensor from shared memory
            result = await self._read_tensor_from_lease(key, lease_info)
            return result
        finally:
            # Step 3: Always release lease
            await self._release_lease(lease_info.lease_id)

    async def _acquire_lease(self, key: CacheEngineKey) -> Optional[LeaseInfo]:
        """Acquire a lease for the given key from Membrain daemon."""
        key_str = self._key_to_string(key)
        url = f"{self.membrain_url}/v1/kv/{self.bucket_name}/{key_str}/leases"
        params = {"timeout_ms": self.timeout_ms}

        result = await self._http_request('POST', url, params=params, timeout=self.timeout_ms/1000.0)

        if result and result['status'] == 200 and result['json']:
            lease_data = result['json']
            return LeaseInfo(
                lease_id=lease_data["id"],
                offsets=[(o["offset"], o["len"]) for o in lease_data["offsets"]],
                total_size=sum(o["len"] for o in lease_data["offsets"])
            )
        return None

    async def _release_lease(self, lease_id: str) -> bool:
        """Release a lease."""
        url = f"{self.membrain_url}/v1/leases/{lease_id}/release"
        result = await self._http_request('POST', url, timeout=2.0)
        return result and result['status'] == 200

    async def _read_tensor_from_lease(self, key: CacheEngineKey, lease_info: LeaseInfo) -> Optional[MemoryObj]:
        """Unified tensor reading from lease - handles both single and multi-block cases."""
        if not await self._ensure_shared_memory():
            return None

        if not lease_info.offsets:
            logger.error(f"No offsets in lease for key {key}")
            return None

        try:
            # Read all data (single block is just multi-block with length=1)
            total_data = bytearray()
            for offset, length in lease_info.offsets:
                chunk = bytes(self.shared_memory_map[offset:offset + length])
                total_data.extend(chunk)

            # Validate total size
            if len(total_data) != lease_info.total_size:
                logger.error(f"Size mismatch: expected {lease_info.total_size}, got {len(total_data)}")
                return None

            # Parse format: [4 bytes metadata size][metadata json][tensor bytes]  
            if len(total_data) < 4:
                logger.error(f"Insufficient data for metadata header")
                return None

            metadata_size = int.from_bytes(total_data[:4], 'little')
            if len(total_data) < 4 + metadata_size:
                logger.error(f"Insufficient data for metadata")
                return None

            # Extract metadata and tensor data
            metadata_json = total_data[4:4 + metadata_size].decode('utf-8')
            tensor_metadata = json.loads(metadata_json)
            tensor_bytes = bytes(total_data[4 + metadata_size:])

            if len(tensor_bytes) != tensor_metadata['tensor_size']:
                logger.error(f"Tensor size mismatch")
                return None

            # Reconstruct tensor using centralized dtype handling
            return await self._create_tensor_from_metadata_and_data(key, tensor_metadata, tensor_bytes)

        except Exception as e:
            logger.error(f"Error reading tensor from lease for key {key}: {e}")
            return None

    async def _create_tensor_from_metadata_and_data(self, key: CacheEngineKey, tensor_metadata: dict, tensor_data) -> Optional[MemoryObj]:
        """Create tensor from metadata and data with minimal copying and centralized dtype handling."""
        try:
            # Parse tensor metadata
            shape = torch.Size(tensor_metadata['shape'])
            original_dtype_str = tensor_metadata['original_dtype']
            serialized_dtype_str = tensor_metadata['serialized_dtype']
            memory_format = MemoryFormat(tensor_metadata['format'])

            # Parse dtype strings safely
            original_dtype = getattr(torch, original_dtype_str.replace('torch.', ''))
            serialized_dtype = getattr(torch, serialized_dtype_str.replace('torch.', ''))

            # Check if this was stored using raw bytes approach
            if DTypeManager.requires_raw_bytes_handling(original_dtype):
                # Handle raw bytes deserialization
                if original_dtype == torch.bfloat16:
                    # Data was stored as uint16, reinterpret back to bfloat16
                    numpy_dtype = np.uint16
                    
                    # Zero-copy numpy array creation
                    numpy_array = np.frombuffer(tensor_data, dtype=numpy_dtype)
                    
                    # Create uint16 tensor first, then reinterpret as bfloat16
                    uint16_tensor = torch.from_numpy(numpy_array).reshape(shape)
                    reconstructed_tensor = uint16_tensor.view(torch.bfloat16)
                    logger.debug(f"Reinterpreted uint16 back to {original_dtype} for raw bytes deserialization")
                else:
                    # Future raw bytes dtypes can be handled here
                    logger.warning(f"Unhandled raw bytes dtype {original_dtype} during deserialization")
                    return None
            else:
                # Standard deserialization path
                # Use centralized dtype manager for numpy conversion
                numpy_dtype = DTypeManager.get_numpy_dtype_from_torch(serialized_dtype)

                # Zero-copy numpy array creation
                numpy_array = np.frombuffer(tensor_data, dtype=numpy_dtype)

                # Create tensor without unnecessary copy
                reconstructed_tensor = torch.from_numpy(numpy_array).reshape(shape)

                # Convert back to original dtype if serialization required conversion
                if original_dtype != serialized_dtype:
                    reconstructed_tensor = reconstructed_tensor.to(original_dtype)
                    logger.debug(f"Converted tensor back from {serialized_dtype} to {original_dtype}")

            # Allocate memory object with correct format
            memory_obj = self.memory_allocator.allocate(shape, original_dtype, memory_format)
            if memory_obj is None:
                logger.error(f"Failed to allocate memory for key {key}")
                return None

            # Efficient device transfer
            if memory_obj.tensor is not None:
                if reconstructed_tensor.device != memory_obj.tensor.device:
                    # Only transfer device if necessary
                    target_tensor = reconstructed_tensor.to(memory_obj.tensor.device, non_blocking=True)
                    memory_obj.tensor.copy_(target_tensor, non_blocking=True)
                else:
                    # Same device - direct copy
                    memory_obj.tensor.copy_(reconstructed_tensor)

                logger.debug(f"Reconstructed tensor: shape={shape}, {serialized_dtype}->{original_dtype}, format={memory_format}")
                return memory_obj
            else:
                logger.error(f"Allocated memory object has no tensor for key {key}")
                memory_obj.ref_count_down()
                return None

        except Exception as e:
            logger.error(f"Error creating tensor from metadata for key {key}: {e}")
            return None

    async def _ensure_shared_memory(self) -> bool:
        """Ensure shared memory is initialized and accessible."""
        with self.shared_memory_lock:
            if self.shared_memory_map is not None:
                return True

            if self.shared_memory_name is None:
                logger.error("No shared memory name configured")
                return False

            try:
                # Try to open existing shared memory segment created by Membrain daemon
                self.shared_memory_obj = shared_memory.SharedMemory(
                    name=self.shared_memory_name, create=False
                )
                self.shared_memory_map = memoryview(self.shared_memory_obj.buf)

                logger.info(f"MembrainBackend: Successfully opened shared memory: {self.shared_memory_name} (size: {len(self.shared_memory_map)} bytes)")
                return True

            except FileNotFoundError:
                logger.error(f"MembrainBackend: CRITICAL - Shared memory segment '{self.shared_memory_name}' not found. Is Membrain daemon running and creating shared memory?")
                return False
            except Exception as e:
                logger.error(f"MembrainBackend: CRITICAL - Failed to initialize shared memory: {e}")
                return False

    def pin(self, key: CacheEngineKey) -> bool:
        """Pin operation - not implemented for Membrain."""
        return True

    def unpin(self, key: CacheEngineKey) -> bool:
        """Unpin operation - not implemented for Membrain."""
        return True

    def close(self) -> None:
        """Close the backend and release resources."""
        # Close HTTP session and connection pool
        if self.http_session is not None:
            try:
                # Schedule closure on the event loop
                asyncio.run_coroutine_threadsafe(
                    self.http_session.close(), self.loop
                ).result(timeout=5.0)
                logger.info("HTTP session closed")
            except Exception as e:
                logger.error(f"Error closing HTTP session: {e}")
            self.http_session = None

        # Shutdown thread pool
        if self.thread_pool is not None:
            try:
                self.thread_pool.shutdown(wait=True, timeout=10.0)
                logger.info("Thread pool shutdown complete")
            except Exception as e:
                logger.error(f"Error shutting down thread pool: {e}")

        # Close shared memory resources
        with self.shared_memory_lock:
            if self.shared_memory_map is not None:
                try:
                    self.shared_memory_map.release()
                except Exception as e:
                    logger.error(f"Error releasing shared memory map: {e}")
                self.shared_memory_map = None

            if self.shared_memory_obj is not None:
                try:
                    self.shared_memory_obj.close()
                except Exception as e:
                    logger.error(f"Error closing shared memory: {e}")
                self.shared_memory_obj = None

        logger.info("MembrainBackend closed with all resources cleaned up.")

    # Helper methods

    def _key_to_string(self, key: CacheEngineKey) -> str:
        """Convert CacheEngineKey to string format for HTTP API.
        
        Use URL encoding for complete safety instead of character replacement.
        This avoids conflicts with existing underscores in keys.
        """
        import urllib.parse
        key_str = key.to_string()
        # URL encode the entire key to handle all special characters safely
        return urllib.parse.quote(key_str, safe="")


    def _memory_obj_to_bytes(self, memory_obj: MemoryObj) -> bytes:
        """Convert MemoryObj to bytes for HTTP transmission with metadata header.
        
        Format: [4 bytes metadata size][metadata json][tensor bytes]
        Uses centralized DTypeManager for consistent dtype handling.
        """
        tensor = memory_obj.tensor
        if tensor is None:
            return b""

        # Store original properties before conversion
        original_shape = tensor.shape
        original_dtype = tensor.dtype
        original_format = memory_obj.get_memory_format()
        
        # Move to CPU if needed
        if tensor.is_cuda:
            tensor = tensor.cpu()
        
        # Apply dtype conversion using centralized manager
        try:
            converted_tensor, serialized_dtype = DTypeManager.apply_serialization_conversion(tensor)
            tensor_bytes = converted_tensor.numpy().tobytes()
        except Exception as e:
            logger.error(f"Failed to convert tensor to bytes, dtype={tensor.dtype}: {e}")
            # Emergency fallback - should rarely happen with proper DTypeManager
            try:
                fallback_tensor = tensor.to(torch.float32)
                tensor_bytes = fallback_tensor.numpy().tobytes()
                serialized_dtype = torch.float32
                logger.warning(f"Used emergency fallback conversion for dtype {original_dtype}")
            except Exception as fallback_error:
                logger.error(f"Emergency fallback also failed: {fallback_error}")
                return b""

        # Create metadata header for proper reconstruction
        metadata_dict = {
            'shape': list(original_shape),
            'original_dtype': str(original_dtype),
            'serialized_dtype': str(serialized_dtype),
            'format': original_format.value,
            'tensor_size': len(tensor_bytes)
        }

        # Serialize metadata as JSON bytes
        metadata_json = json.dumps(metadata_dict).encode('utf-8')
        metadata_size = len(metadata_json)

        # Format: [4 bytes metadata size][metadata json][tensor bytes]
        result = metadata_size.to_bytes(4, 'little') + metadata_json + tensor_bytes

        logger.debug(f"Serialized tensor: shape={original_shape}, {original_dtype}->{serialized_dtype}, size={len(result)} bytes")
        return result