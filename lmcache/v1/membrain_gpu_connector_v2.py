"""
Streamlined zero-copy GPU connector for Membrain backend.
Focuses on direct, efficient data transfer without redundancy.
"""

import logging
from typing import List, Optional
import torch
import lmcache.c_ops as lmc_ops
from lmcache.utils import _lmcache_nvtx_annotate
from lmcache.v1.gpu_connector import VLLMPagedMemLayerwiseGPUConnector
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.shared_memory_manager import SharedMemoryManager
from lmcache.v1.membrain_parser import MembrainDataParser

logger = logging.getLogger(__name__)


class BedrockMembrainGPUConnectorV2(VLLMPagedMemLayerwiseGPUConnector):
    """
    Streamlined zero-copy GPU connector for Membrain shared memory.
    
    Features:
    - Single SharedMemoryManager instance for all operations
    - Unified parsing logic through MembrainDataParser
    - Direct zero-copy path without intermediate tensors
    - Simplified error handling and logging
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize with centralized shared memory management."""
        super().__init__(*args, **kwargs)
        self._shm_manager = SharedMemoryManager('membrain-kvcache')
        self._parser = MembrainDataParser()
        
        # Pre-attach to shared memory for better performance
        if self._shm_manager.attach():
            logger.info("✅ Pre-attached to Membrain shared memory")
        else:
            logger.warning("Could not pre-attach to shared memory")
    
    @staticmethod
    def from_base(conn: VLLMPagedMemLayerwiseGPUConnector) -> "BedrockMembrainGPUConnectorV2":
        """Convert base connector to Membrain connector."""
        conn.__class__ = BedrockMembrainGPUConnectorV2
        # Initialize Membrain-specific attributes
        conn._shm_manager = SharedMemoryManager('membrain-kvcache')
        conn._parser = MembrainDataParser()
        conn._shm_manager.attach()
        return conn
    
    def get_shape(self, num_tokens: int) -> torch.Size:
        """Get expected tensor shape for given number of tokens."""
        return torch.Size([num_tokens, 2, self.hidden_dim_size])
    
    @_lmcache_nvtx_annotate
    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """
        Handle direct zero-copy transfer to GPU for non-layerwise operations.
        
        Args:
            memory_obj: Memory object containing lease offsets
            start: Start token index
            end: End token index
            **kwargs: Must contain 'kvcaches' and 'slot_mapping'
        """
        # Fallback to parent for non-Membrain objects
        if not hasattr(memory_obj, 'offsets'):
            return super().to_gpu(memory_obj, start, end, **kwargs)
        
        # Validate required arguments
        kvcaches = kwargs.get("kvcaches")
        slot_mapping = kwargs.get("slot_mapping")
        if kvcaches is None or slot_mapping is None:
            raise ValueError("'kvcaches' and 'slot_mapping' required")
        
        num_tokens = end - start
        
        # Get shared memory buffer
        shm_buffer = self._shm_manager.get_buffer()
        if shm_buffer is None:
            raise RuntimeError("Failed to access shared memory")
        
        # Parse data directly from shared memory
        kv_tensor, metadata = self._parser.parse(
            memory_obj.offsets,
            shm_buffer,
            num_tokens,
            dtype=getattr(self, 'dtype', torch.float16),
            hidden_dim=self.hidden_dim_size
        )
        
        if kv_tensor is None:
            raise ValueError("Failed to parse Membrain data")
        
        # Transfer to GPU
        self._transfer_layers_to_gpu(
            kv_tensor, kvcaches, slot_mapping, 
            num_tokens, start, end
        )
    
    def _transfer_layers_to_gpu(
        self, kv_tensor: torch.Tensor, kvcaches: List,
        slot_mapping: torch.Tensor, num_tokens: int,
        start: int, end: int
    ):
        """
        Transfer parsed tensor data to GPU layers.
        
        Args:
            kv_tensor: Flattened KV tensor data
            kvcaches: List of KV cache tensors per layer
            slot_mapping: GPU memory slot mapping
            num_tokens: Number of tokens to transfer
            start: Start index in slot mapping
            end: End index in slot mapping
        """
        elements_per_layer = num_tokens * 2 * self.hidden_dim_size
        num_layers_in_data = len(kv_tensor) // elements_per_layer
        effective_layers = min(num_layers_in_data, self.num_layers)
        
        logger.debug(f"Transferring {effective_layers} layers to GPU")
        
        for layer_id in range(effective_layers):
            # Extract layer data
            layer_start = layer_id * elements_per_layer
            layer_end = layer_start + elements_per_layer
            layer_tensor = kv_tensor[layer_start:layer_end]
            
            # Reshape and transfer to GPU
            layer_tensor = layer_tensor.view(num_tokens, 2, self.hidden_dim_size)
            gpu_tensor = layer_tensor.to(
                device=kvcaches[layer_id][0].device, 
                non_blocking=True
            )
            
            # Transfer to vLLM KV cache
            lmc_ops.single_layer_kv_transfer(
                gpu_tensor,
                kvcaches[layer_id][0],
                kvcaches[layer_id][1],
                slot_mapping[start:end],
                False,  # direction: LMCache -> vLLM
                True,   # token_major format
            )
    
    @_lmcache_nvtx_annotate
    def batched_to_gpu(self, starts: List[int], ends: List[int], **kwargs):
        """
        Generator for layerwise KV cache loading with chunk-aware processing.
        
        Args:
            starts: List of start indices for each chunk
            ends: List of end indices for each chunk
            **kwargs: Must contain 'kvcaches' and 'slot_mapping'
        """
        kvcaches = kwargs.get("kvcaches")
        slot_mapping = kwargs.get("slot_mapping")
        
        if kvcaches is None or slot_mapping is None:
            raise ValueError("'kvcaches' and 'slot_mapping' required")
        
        # Get shared memory buffer once
        shm_buffer = self._shm_manager.get_buffer()
        if shm_buffer is None:
            raise RuntimeError("Failed to access shared memory")
        
        # Prepare slot mappings for chunks
        slot_mapping_chunks = [
            slot_mapping[start:end] 
            for start, end in zip(starts, ends, strict=False)
        ]
        
        # Allocate GPU buffer for compatibility
        num_tokens = sum(end - start for start, end in zip(starts, ends))
        buffer_shape = torch.Size([num_tokens * 2 * self.hidden_dim_size])
        tmp_gpu_buffer = self.gpu_buffer_allocator.allocate(
            buffer_shape, torch.int8, MemoryFormat.KV_T2D
        )
        
        current_stream = torch.cuda.current_stream()
        
        try:
            # Process each layer
            for layer_id in range(self.num_layers):
                memory_objs_layer = yield
                
                if layer_id > 0:
                    current_stream.wait_stream(self.load_stream)
                
                with torch.cuda.stream(self.load_stream):
                    # Process each chunk for this layer
                    cumulative_offset = 0
                    
                    for chunk_idx, (start, end, memory_obj) in enumerate(
                        zip(starts, ends, memory_objs_layer, strict=False)
                    ):
                        if memory_obj is None or not hasattr(memory_obj, 'offsets'):
                            logger.warning(f"Skipping invalid memory object at chunk {chunk_idx}")
                            continue
                        
                        chunk_tokens = end - start
                        chunk_slot_mapping = slot_mapping_chunks[chunk_idx]
                        
                        # Parse and transfer this chunk
                        self._process_chunk(
                            memory_obj, shm_buffer, chunk_tokens,
                            kvcaches, layer_id, chunk_slot_mapping
                        )
                        
                        cumulative_offset += chunk_tokens
            
            # Final synchronization
            yield
            current_stream.wait_stream(self.load_stream)
            yield
            
        finally:
            tmp_gpu_buffer.ref_count_down()
    
    def _process_chunk(
        self, memory_obj: MemoryObj, shm_buffer: memoryview,
        chunk_tokens: int, kvcaches: List, layer_id: int,
        chunk_slot_mapping: torch.Tensor
    ):
        """
        Process a single chunk for a specific layer.
        
        Args:
            memory_obj: Memory object with offsets
            shm_buffer: Shared memory buffer
            chunk_tokens: Number of tokens in this chunk
            kvcaches: KV cache tensors
            layer_id: Current layer ID
            chunk_slot_mapping: Slot mapping for this chunk
        """
        # Get actual token count from metadata
        metadata_shape = memory_obj.metadata.shape
        actual_tokens = metadata_shape[0]
        
        # Parse chunk data
        kv_tensor, metadata = self._parser.parse(
            memory_obj.offsets,
            shm_buffer,
            actual_tokens,
            dtype=getattr(self, 'dtype', torch.float16),
            hidden_dim=self.hidden_dim_size
        )
        
        if kv_tensor is None:
            raise ValueError(f"Failed to parse chunk data for layer {layer_id}")
        
        # Transfer to GPU
        gpu_tensor = kv_tensor.to(
            device=kvcaches[layer_id][0].device,
            non_blocking=True
        )
        
        # Reshape and transfer
        reshaped_tensor = gpu_tensor.view(actual_tokens, 2, self.hidden_dim_size)
        
        lmc_ops.single_layer_kv_transfer(
            reshaped_tensor,
            kvcaches[layer_id][0],
            kvcaches[layer_id][1],
            chunk_slot_mapping,
            False,  # direction: LMCache -> vLLM
            True,   # token_major format
        )
        
        logger.debug(f"Transferred chunk to layer {layer_id}: {actual_tokens} tokens")
    
    def __del__(self):
        """Cleanup shared memory manager on deletion."""
        if hasattr(self, '_shm_manager'):
            self._shm_manager.close()