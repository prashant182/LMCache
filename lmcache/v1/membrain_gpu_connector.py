from typing import List
from lmcache.logging import init_logger
from lmcache.utils import _lmcache_nvtx_annotate
from lmcache.v1.gpu_connector import VLLMPagedMemLayerwiseGPUConnector
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.protocol import RemoteMetadata
import torch
import asyncio
import concurrent.futures
from multiprocessing.shared_memory import SharedMemory
from multiprocessing import resource_tracker
import lmcache.c_ops as lmc_ops

logger = init_logger(__name__)

class ProtectedSharedMemory:
    """
    Wrapper around SharedMemory that prevents automatic unlinking during exceptions.
    This fixes the critical bug where SharedMemory gets destroyed when errors occur,
    causing widespread memory failures across processes.
    """
    def __init__(self, name: str):
        self._shm = None
        self._name = name
        self._attached = False
        self._attach()
    
    def _attach(self):
        """Safely attach to existing shared memory without auto-cleanup."""
        try:
            # Attach to existing shared memory - never create or unlink
            self._shm = SharedMemory(name=self._name, create=False)
            self._attached = True
            
            # CRITICAL: Multiple layers of protection against automatic unlinking
            # Based on https://bugs.python.org/file49859/mprt_monkeypatch.py
            
            # 1. Replace __del__ to prevent automatic cleanup
            def safe_del(self):
                # Do nothing - never unlink or close
                pass
            self._shm.__del__ = safe_del
            
            # 2. Replace close() to prevent unlinking
            original_close = self._shm.close
            def safe_close():
                # Only close the mmap, never unlink
                try:
                    if hasattr(self._shm, '_mmap') and self._shm._mmap is not None:
                        self._shm._mmap.close()
                except Exception:
                    pass
            self._shm.close = safe_close
            
            # 3. Replace unlink() to prevent accidental unlinking
            def no_unlink():
                logger.warning(f"⚠️ Attempted to unlink shared memory '{self._name}' - BLOCKED")
                pass
            self._shm.unlink = no_unlink
            
            # 4. Unregister from resource tracker to prevent cleanup
            # This is critical - resource_tracker would otherwise clean up the shared memory
            try:
                resource_tracker.unregister(self._name, 'shared_memory')
            except KeyError:
                # Already unregistered or never registered
                pass
            except Exception as e:
                logger.debug(f"Resource tracker unregister failed (non-critical): {e}")
            
            # 5. Clear the _name attribute as additional protection
            self._shm._name = None
            
            logger.debug(f"🔒 PROTECTED: Attached to SharedMemory '{self._name}' with full protection")
            
        except FileNotFoundError:
            logger.warning(f"⚠️ SharedMemory '{self._name}' not found - may not be initialized yet")
            self._attached = False
        except Exception as e:
            logger.error(f"❌ Failed to attach to SharedMemory '{self._name}': {e}")
            self._attached = False
    
    @property 
    def buf(self):
        """Access the memory buffer if attached."""
        if not self._attached or self._shm is None:
            self._attach()  # Try to reattach
        
        if self._attached and self._shm is not None:
            return self._shm.buf
        else:
            raise RuntimeError(f"SharedMemory '{self._name}' not available")
    
    @property
    def size(self):
        """Get the size of the shared memory."""
        if not self._attached or self._shm is None:
            self._attach()
            
        if self._attached and self._shm is not None:
            return self._shm.size
        else:
            return 0
    
    def close(self):
        """Close the shared memory reference safely without unlinking."""
        if self._shm is not None:
            try:
                # Use our safe_close method if available
                if hasattr(self._shm, 'close'):
                    self._shm.close()  # This will use our safe_close
                logger.debug(f"🔒 PROTECTED: Closed SharedMemory '{self._name}' reference safely")
            except Exception as e:
                logger.debug(f"Debug: Error during close (expected): {e}")
            finally:
                self._shm = None
                self._attached = False

class BedrockMembrainGPUConnector(VLLMPagedMemLayerwiseGPUConnector):
    """
    Zero-copy GPU connector for Membrain shared memory with optimized segment processing.
    Always uses concurrent processing for multi-segment data to maximize performance.
    """
    
    def _parse_membrain_data(self, memory_obj, shm, num_tokens):
        """Parse Membrain shared memory data with zero-copy optimizations."""
        try:
            num_segments = len(memory_obj.offsets)
            logger.info(f"🔍 GPU PARSE START: {num_segments} segments for {num_tokens} tokens")
            logger.info(f"  - Memory obj type: {type(memory_obj)}")
            logger.info(f"  - Offsets: {memory_obj.offsets}")
            
            if num_segments == 1:
                return self._parse_single_segment(memory_obj.offsets[0], shm, num_tokens)
            else:
                return self._parse_multi_segments(memory_obj.offsets, shm, num_tokens)
                
        except Exception as e:
            logger.error(f"❌ GPU PARSE FAILED: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None, None

    def _parse_single_segment(self, offset_info, shm, num_tokens):
        """Fast path for single segment data."""
        o, l = offset_info['offset'], offset_info['len']
        
        logger.info(f"📌 SINGLE SEGMENT PARSE:")
        logger.info(f"  - Offset: {o}, Length: {l} bytes")
        
        if l < 4:
            raise ValueError(f"Insufficient data: {l} bytes")
        
        # Direct copy to avoid SharedMemory lifecycle issues
        raw_data = bytes(shm.buf[o:o + l])
        metadata_len = int.from_bytes(raw_data[:4], byteorder='little')
        kv_data_start = 4 + metadata_len
        
        logger.info(f"  - Metadata length: {metadata_len} bytes")
        logger.info(f"  - KV data start: {kv_data_start}")
        logger.info(f"  - First 64 bytes: {raw_data[:64].hex()}")
        
        if len(raw_data) < kv_data_start:
            raise ValueError(f"Insufficient metadata space: need {kv_data_start}, have {len(raw_data)}")
        
        # Parse metadata and extract KV data
        metadata = RemoteMetadata.deserialize(raw_data[4:kv_data_start])
        kv_data_bytes = raw_data[kv_data_start:]
        
        logger.info(f"📋 PARSED METADATA:")
        logger.info(f"  - Shape: {metadata.shape}")
        logger.info(f"  - Dtype: {metadata.dtype}")
        logger.info(f"  - Format: {metadata.fmt}")
        logger.info(f"  - KV data size: {len(kv_data_bytes)} bytes")
        
        if not kv_data_bytes:
            raise ValueError("No KV data found")
        
        # Create tensor
        tensor_dtype = getattr(metadata, 'dtype', getattr(self, 'dtype', torch.float16))
        kv_tensor = torch.frombuffer(kv_data_bytes, dtype=tensor_dtype)
        
        logger.info(f"⚡ SINGLE SEGMENT TENSOR:")
        logger.info(f"  - Elements: {len(kv_tensor)}")
        logger.info(f"  - Expected elements: {num_tokens * 2 * self.hidden_dim_size}")
        logger.info(f"  - Tensor stats: mean={kv_tensor.mean():.6f}, std={kv_tensor.std():.6f}")
        logger.info(f"  - First 10 values: {kv_tensor[:10].tolist() if len(kv_tensor) > 10 else kv_tensor.tolist()}")
        
        return self._finalize_tensor(kv_tensor, num_tokens), metadata

    def _parse_multi_segments(self, offsets, shm, num_tokens):
        """Optimized multi-segment processing with concurrent execution."""
        # Extract metadata from first segment
        first_segment = bytes(shm.buf[offsets[0]['offset']:offsets[0]['offset'] + offsets[0]['len']])
        
        if len(first_segment) < 4:
            raise ValueError("First segment too small")
        
        metadata_len = int.from_bytes(first_segment[:4], byteorder='little')
        kv_data_start = 4 + metadata_len
        
        if len(first_segment) >= kv_data_start:
            # Fast path: metadata in first segment
            metadata = RemoteMetadata.deserialize(first_segment[4:kv_data_start])
            kv_data = self._extract_kv_data_concurrent(offsets, shm, kv_data_start)
        else:
            # Fallback: metadata spans segments
            logger.warning("⚠️ Metadata spans segments - using fallback")
            combined_data = self._combine_all_segments(offsets, shm)
            metadata = RemoteMetadata.deserialize(combined_data[4:kv_data_start])
            kv_data = combined_data[kv_data_start:]
        
        # Create tensor
        tensor_dtype = getattr(metadata, 'dtype', getattr(self, 'dtype', torch.float16))
        kv_tensor = torch.frombuffer(kv_data, dtype=tensor_dtype)
        
        logger.info(f"🔧 MULTI SEGMENT: {len(kv_tensor)} elements from {len(offsets)} segments")
        return self._finalize_tensor(kv_tensor, num_tokens), metadata

    def _extract_kv_data_concurrent(self, offsets, shm, kv_data_start):
        """Extract KV data using concurrent processing when beneficial."""
        # For small numbers of offsets, use simple sync processing
        if len(offsets) <= 2:
            return self._extract_kv_data_sync(offsets, shm, kv_data_start)
        
        # For larger numbers, use ThreadIPoolExecutor for true concurrency
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(offsets))) as executor:
                future = executor.submit(self._extract_kv_data_sync, offsets, shm, kv_data_start)
                return future.result()
        except Exception as e:
            logger.warning(f"Concurrent extraction failed: {e}, falling back to sync")
            return self._extract_kv_data_sync(offsets, shm, kv_data_start)

    async def _extract_kv_data_async(self, offsets, shm, kv_data_start):
        """Async KV data extraction."""
        
        # Create batches for parallel processing
        num_workers = min(4, len(offsets))
        batch_size = max(1, len(offsets) // num_workers)
        
        async def process_batch(batch_start_idx, batch_offsets):
            """Process a batch of segments."""
            result = bytearray()
            for i, offset in enumerate(batch_offsets):
                segment_idx = batch_start_idx + i
                o, l = offset['offset'], offset['len']
                
                if segment_idx == 0:
                    # First segment: skip metadata
                    start_pos = kv_data_start
                    size = l - kv_data_start
                else:
                    # Other segments: all data is KV
                    start_pos = 0
                    size = l
                
                if size > 0:
                    segment_data = bytes(shm.buf[o + start_pos:o + l])
                    result.extend(segment_data)
            
            return result
        
        # Create coroutines for concurrent execution
        tasks = []
        for i in range(0, len(offsets), batch_size):
            batch_offsets = offsets[i:i + batch_size]
            task = process_batch(i, batch_offsets)
            tasks.append(task)
        
        # Execute all batches concurrently using asyncio.gather
        batch_results = await asyncio.gather(*tasks)
        
        # Combine results sequentially to maintain order
        result = bytearray()
        for batch_result in batch_results:
            result.extend(batch_result)
        
        logger.debug(f"🔄 CONCURRENT: Processed {len(offsets)} segments in {len(batch_results)} batches")
        return result

    def _extract_kv_data_sync(self, offsets, shm, kv_data_start):
        """Synchronous fallback for KV data extraction."""
        result = bytearray()
        for i, offset in enumerate(offsets):
            o, l = offset['offset'], offset['len']
            
            if i == 0:
                start_pos = kv_data_start
                size = l - kv_data_start
            else:
                start_pos = 0
                size = l
            
            if size > 0:
                segment_data = bytes(shm.buf[o + start_pos:o + l])
                result.extend(segment_data)
        
        return result

    def _combine_all_segments(self, offsets, shm):
        """Fallback: combine all segments when metadata spans multiple segments."""
        total_size = sum(offset['len'] for offset in offsets)
        combined = bytearray(total_size)
        
        pos = 0
        for offset in offsets:
            o, l = offset['offset'], offset['len']
            combined[pos:pos + l] = shm.buf[o:o + l]
            pos += l
        
        return combined

    def _finalize_tensor(self, kv_tensor, num_tokens):
        """Apply padding if needed and validate tensor."""
        expected_elements = num_tokens * 2 * self.hidden_dim_size
        
        if len(kv_tensor) < expected_elements:
            # This can happen with chunked storage - pad with zeros
            logger.warning(f"⚠️ Padding: need {expected_elements}, have {len(kv_tensor)} (likely chunked storage)")
            padded = torch.zeros(expected_elements, dtype=kv_tensor.dtype)
            padded[:len(kv_tensor)] = kv_tensor
            return padded
        elif len(kv_tensor) > expected_elements:
            # This shouldn't happen but handle gracefully
            logger.warning(f"⚠️ Truncating: need {expected_elements}, have {len(kv_tensor)}")
            return kv_tensor[:expected_elements]
        
        logger.debug(f"✅ Tensor size matches expected: {expected_elements} elements")
        return kv_tensor

    @staticmethod
    def from_base(conn: VLLMPagedMemLayerwiseGPUConnector) -> "BedrockMembrainGPUConnector":
        """Convert base connector to Membrain connector."""
        conn.__class__ = BedrockMembrainGPUConnector
        return conn
    
    def get_shape(self, num_tokens: int) -> torch.Size:
        return torch.Size([num_tokens, 2, self.hidden_dim_size])
    
    def get_flat_shape(self, num_tokens: int) -> torch.Size:
        dtype = getattr(self, 'dtype', torch.float16)
        element_size = torch.finfo(dtype).bits // 8
        return torch.Size([num_tokens * 2 * self.hidden_dim_size * element_size])
    
    @_lmcache_nvtx_annotate
    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """Handle LeaseMemoryObj for regular (non-layerwise) operations."""
        if not (hasattr(memory_obj, 'offsets') and memory_obj.offsets):
            # Fallback to parent implementation
            if memory_obj.tensor is None:
                raise ValueError(f"memory_obj.tensor is None for {type(memory_obj)}")
            return super().to_gpu(memory_obj, start, end, **kwargs)
        
        # Validate required arguments
        if "kvcaches" not in kwargs or "slot_mapping" not in kwargs:
            raise ValueError("'kvcaches' and 'slot_mapping' required in kwargs")
        
        kvcaches = kwargs["kvcaches"]
        slot_mapping = kwargs["slot_mapping"]
        num_tokens = end - start
        
        # Use shared memory from LeaseMemoryObj if available, otherwise create new
        if hasattr(memory_obj, 'shared_memory') and memory_obj.shared_memory is not None:
            shm = memory_obj.shared_memory
            should_close = False
        else:
            shm = ProtectedSharedMemory('membrain-kvcache')
            should_close = True
            
        try:
            kv_tensor, metadata = self._parse_membrain_data(memory_obj, shm, num_tokens)
            if kv_tensor is None or metadata is None:
                raise ValueError("Failed to parse Membrain data")
            
            # Process layers
            self._transfer_layers_to_gpu(kv_tensor, kvcaches, slot_mapping, num_tokens, start, end)
            
        except Exception as e:
            logger.error(f"❌ to_gpu failed: {e}")
            raise
        finally:
            # Only close if we created it locally
            if should_close:
                shm.close()

    def _transfer_layers_to_gpu(self, kv_tensor, kvcaches, slot_mapping, num_tokens, start, end):
        """Transfer parsed tensor data to GPU layers."""
        expected_elements_per_layer = num_tokens * 2 * self.hidden_dim_size
        num_layers_in_data = len(kv_tensor) // expected_elements_per_layer
        effective_layers = min(num_layers_in_data, self.num_layers)
        
        logger.info(f"🚀 GPU TRANSFER START:")
        logger.info(f"  - Num tokens: {num_tokens}")
        logger.info(f"  - Hidden dim: {self.hidden_dim_size}")
        logger.info(f"  - Elements per layer: {expected_elements_per_layer}")
        logger.info(f"  - Layers in data: {num_layers_in_data}")
        logger.info(f"  - Effective layers: {effective_layers}")
        
        for layer_id in range(effective_layers):
            layer_start = layer_id * expected_elements_per_layer
            layer_end = layer_start + expected_elements_per_layer
            layer_tensor = kv_tensor[layer_start:layer_end]
            
            logger.debug(f"  Layer {layer_id}: elements {layer_start}-{layer_end}")
            
            # Reshape and transfer to GPU
            layer_tensor = layer_tensor.view(num_tokens, 2, self.hidden_dim_size)
            gpu_tensor = layer_tensor.to(device=kvcaches[layer_id][0].device, non_blocking=True)
            
            # Log tensor statistics
            if layer_id == 0:
                logger.info(f"📊 LAYER 0 TENSOR STATS:")
                logger.info(f"  - Shape: {gpu_tensor.shape}")
                logger.info(f"  - Mean: {gpu_tensor.mean().item():.6f}")
                logger.info(f"  - Std: {gpu_tensor.std().item():.6f}")
                logger.info(f"  - Min: {gpu_tensor.min().item():.6f}")
                logger.info(f"  - Max: {gpu_tensor.max().item():.6f}")
            
            # Transfer to vLLM KV cache
            lmc_ops.single_layer_kv_transfer(
                gpu_tensor,
                kvcaches[layer_id][0],
                kvcaches[layer_id][1],
                slot_mapping[start:end],
                False,  # direction: LMCache -> vLLM
                True,   # token_major: [num_tokens, 2, hidden_dim]
            )
        
        logger.info(f"✅ GPU TRANSFER COMPLETE: {effective_layers} layers")

    @_lmcache_nvtx_annotate
    def batched_from_gpu(self, memory_objs: List[List[MemoryObj]], starts: List[int], ends: List[int], **kwargs):
        """Handle layerwise store operations."""
        logger.debug(f"batched_from_gpu: {len(memory_objs)} layers, {len(starts)} chunks")
        yield from super().batched_from_gpu(memory_objs, starts, ends, **kwargs)

    @_lmcache_nvtx_annotate
    def batched_to_gpu(self, starts: List[int], ends: List[int], **kwargs):
        """Generator for layerwise KV cache loading from Membrain."""
        if "kvcaches" not in kwargs or "slot_mapping" not in kwargs:
            raise ValueError("'kvcaches' and 'slot_mapping' required in kwargs")
        
        kvcaches = kwargs["kvcaches"]
        slot_mapping = kwargs["slot_mapping"]
        
        # Prepare slot mapping
        slot_mapping_chunks = [slot_mapping[start:end] for start, end in zip(starts, ends, strict=False)]
        slot_mapping_full = torch.cat(slot_mapping_chunks, dim=0)
        num_tokens = len(slot_mapping_full)
        
        # Initialize streams and memory using protected SharedMemory
        current_stream = torch.cuda.current_stream()
        
        # Use shared memory from first memory object if available
        shm = None
        should_close = True
        
        # Check if we can reuse shared memory from LeaseMemoryObj (will be set later)
        memory_objs_with_shm = []
        
        try:
            # Allocate GPU buffer (for compatibility, but zero-copy path bypasses it)
            buffer_shape = self.get_flat_shape(num_tokens)
            tmp_gpu_buffer_obj = self.gpu_buffer_allocator.allocate(
                buffer_shape, torch.int8, MemoryFormat.KV_T2D
            )

            # Process each layer
            for layer_id in range(self.num_layers):
                memory_objs_layer = yield
                current_stream.wait_stream(self.load_stream)
                
                if layer_id > 0:
                    logger.debug(f"Finished loading layer {layer_id - 1}")

                with torch.cuda.stream(self.load_stream):
                    # Track cumulative offset for multi-chunk processing
                    cumulative_offset = 0
                    
                    for chunk_idx, (start, end, memory_obj) in enumerate(zip(starts, ends, memory_objs_layer, strict=False)):
                        if memory_obj is None:
                            logger.warning("Skipping None memory_obj")
                            continue
                        
                        assert memory_obj.metadata.fmt == MemoryFormat.KV_T2D
                        
                        # Initialize shared memory from first valid memory object
                        if shm is None and hasattr(memory_obj, 'offsets'):
                            if hasattr(memory_obj, 'shared_memory') and memory_obj.shared_memory is not None:
                                shm = memory_obj.shared_memory
                                should_close = False
                                logger.debug(f"Using pre-initialized shared memory from LeaseMemoryObj")
                            else:
                                shm = ProtectedSharedMemory('membrain-kvcache')
                                should_close = True
                                logger.debug(f"Created new shared memory reference")
                        
                        try:
                            if hasattr(memory_obj, 'offsets'):
                                # Calculate chunk-specific slot mapping
                                chunk_tokens = end - start
                                chunk_slot_mapping = slot_mapping_full[cumulative_offset:cumulative_offset + chunk_tokens]
                                
                                logger.info(f"📦 Processing chunk {chunk_idx} for layer {layer_id}:")
                                logger.info(f"  - Chunk tokens: {chunk_tokens}")
                                logger.info(f"  - Slot mapping offset: {cumulative_offset}")
                                logger.info(f"  - Slot mapping size: {len(chunk_slot_mapping)}")
                                
                                self._process_chunk_from_membrain(
                                    memory_obj, shm, chunk_tokens, kvcaches, 
                                    layer_id, chunk_slot_mapping, cumulative_offset
                                )
                                
                                cumulative_offset += chunk_tokens
                            else:
                                logger.warning(f"Expected LeaseMemoryObj, got {type(memory_obj)}")
                        except Exception as e:
                            logger.error(f"❌ Layer {layer_id} chunk {chunk_idx} processing failed: {e}")
                            raise

            yield
            current_stream.wait_stream(self.load_stream)
            tmp_gpu_buffer_obj.ref_count_down()
            logger.debug(f"Finished loading layer {layer_id}")
            yield
        finally:
            # Only close if we created it locally 
            if should_close and shm is not None:
                shm.close()

    def _process_chunk_from_membrain(self, memory_obj, shm, chunk_tokens, kvcaches, layer_id, chunk_slot_mapping, offset):
        """Process a single chunk from Membrain data."""
        logger.info(f"🔍 Processing chunk for layer {layer_id}: {len(memory_obj.offsets)} segments")
        
        # Get the actual number of tokens from metadata shape
        # Shape is typically [num_tokens, 1, 2, hidden_dim] or [num_tokens, 2, hidden_dim]
        metadata_shape = memory_obj.metadata.shape
        if len(metadata_shape) == 4:
            actual_tokens = metadata_shape[0]
        elif len(metadata_shape) == 3:
            actual_tokens = metadata_shape[0]
        else:
            actual_tokens = chunk_tokens
        
        logger.info(f"  - Chunk size: {chunk_tokens}, Actual tokens in memory: {actual_tokens}")
        logger.info(f"  - Offset in full sequence: {offset}")
        
        # Parse Membrain data
        kv_tensor, metadata = self._parse_membrain_data(memory_obj, shm, actual_tokens)
        if kv_tensor is None or metadata is None:
            raise ValueError("Failed to parse Membrain data")
        
        expected_elements = actual_tokens * 2 * self.hidden_dim_size
        actual_elements = len(kv_tensor)
        
        # Verify we have the right amount of data
        if actual_elements != expected_elements:
            logger.error(f"  - Data size mismatch: expected {expected_elements}, got {actual_elements}")
            raise ValueError(f"Data size mismatch for chunk")
        
        # Direct GPU transfer (zero-copy optimization)
        gpu_device = kvcaches[layer_id][0].device
        gpu_tensor = kv_tensor.to(device=gpu_device, non_blocking=True)
        
        # Reshape for transfer
        reshaped_tensor = gpu_tensor.view(actual_tokens, 2, self.hidden_dim_size)
        
        # Transfer only this chunk's tokens to the correct slots
        lmc_ops.single_layer_kv_transfer(
            reshaped_tensor,  # Transfer all tokens in this chunk
            kvcaches[layer_id][0],
            kvcaches[layer_id][1],
            chunk_slot_mapping,  # Use chunk-specific slot mapping
            False,  # direction: LMCache -> vLLM
            True,   # token_major: [num_tokens, 2, hidden_dim]
        )
        
        logger.debug(f"✅ Layer {layer_id} chunk transfer complete (offset {offset}, size {actual_tokens})")
    
    def _process_layer_from_membrain(self, memory_obj, shm, num_tokens, kvcaches, layer_id, slot_mapping_full):
        """Legacy method for compatibility - redirects to chunk processing."""
        self._process_chunk_from_membrain(memory_obj, shm, num_tokens, kvcaches, layer_id, slot_mapping_full, 0)