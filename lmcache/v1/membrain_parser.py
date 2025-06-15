"""
Unified parser for Membrain shared memory data.
Simplifies the parsing logic into a single, efficient implementation.
"""

import logging
from typing import List, Tuple, Optional, Dict
import torch
from lmcache.v1.protocol import RemoteMetadata

logger = logging.getLogger(__name__)


class MembrainDataParser:
    """
    Unified parser for Membrain shared memory data.
    Handles both single and multi-segment data efficiently.
    """
    
    @staticmethod
    def parse(
        offsets: List[Dict[str, int]], 
        shm_buffer: memoryview,
        expected_tokens: int,
        dtype: torch.dtype = torch.float16,
        hidden_dim: int = None
    ) -> Tuple[Optional[torch.Tensor], Optional[RemoteMetadata]]:
        """
        Parse Membrain data from shared memory segments.
        
        Args:
            offsets: List of offset dictionaries with 'offset' and 'len' keys
            shm_buffer: Shared memory buffer view
            expected_tokens: Expected number of tokens
            dtype: Expected tensor data type
            hidden_dim: Hidden dimension size (for validation)
            
        Returns:
            Tuple of (kv_tensor, metadata) or (None, None) on failure
        """
        if not offsets or not shm_buffer:
            logger.error("Invalid inputs: empty offsets or buffer")
            return None, None
        
        try:
            # 1. Extract and parse metadata from first segment
            metadata, kv_data_start = MembrainDataParser._parse_metadata(
                offsets[0], shm_buffer
            )
            if not metadata:
                return None, None
            
            # 2. Extract KV data from all segments
            kv_data = MembrainDataParser._extract_kv_data(
                offsets, shm_buffer, kv_data_start
            )
            if not kv_data:
                return None, None
            
            # 3. Create tensor from KV data
            kv_tensor = torch.frombuffer(
                kv_data, 
                dtype=metadata.dtype if hasattr(metadata, 'dtype') else dtype
            )
            
            # 4. Validate and possibly pad tensor
            kv_tensor = MembrainDataParser._validate_tensor(
                kv_tensor, expected_tokens, hidden_dim or metadata.shape[-1]
            )
            
            logger.debug(f"Successfully parsed {len(kv_tensor)} elements from {len(offsets)} segments")
            return kv_tensor, metadata
            
        except Exception as e:
            logger.error(f"Failed to parse Membrain data: {e}")
            return None, None
    
    @staticmethod
    def _parse_metadata(
        first_offset: Dict[str, int], 
        shm_buffer: memoryview
    ) -> Tuple[Optional[RemoteMetadata], int]:
        """
        Parse metadata from the first segment.
        
        Returns:
            Tuple of (metadata, kv_data_start_position)
        """
        offset = first_offset['offset']
        length = first_offset['len']
        
        if length < 4:
            logger.error(f"First segment too small: {length} bytes")
            return None, 0
        
        # Read metadata length (4 bytes)
        metadata_len = int.from_bytes(
            shm_buffer[offset:offset + 4], 
            byteorder='little'
        )
        
        if metadata_len <= 0 or metadata_len > length - 4:
            logger.error(f"Invalid metadata length: {metadata_len}")
            return None, 0
        
        # Read and deserialize metadata
        metadata_start = offset + 4
        metadata_end = metadata_start + metadata_len
        
        try:
            metadata_bytes = shm_buffer[metadata_start:metadata_end]
            metadata = RemoteMetadata.deserialize(metadata_bytes)
            
            # Calculate where KV data starts (relative to first segment)
            kv_data_start = 4 + metadata_len
            
            logger.debug(f"Parsed metadata: shape={metadata.shape}, dtype={metadata.dtype}")
            return metadata, kv_data_start
            
        except Exception as e:
            logger.error(f"Failed to deserialize metadata: {e}")
            return None, 0
    
    @staticmethod
    def _extract_kv_data(
        offsets: List[Dict[str, int]], 
        shm_buffer: memoryview,
        kv_data_start: int
    ) -> Optional[bytes]:
        """
        Extract KV data from all segments efficiently.
        
        Args:
            offsets: List of segment offsets
            shm_buffer: Shared memory buffer
            kv_data_start: Position where KV data starts in first segment
            
        Returns:
            Combined KV data as bytes
        """
        result = bytearray()
        
        for i, offset in enumerate(offsets):
            seg_offset = offset['offset']
            seg_length = offset['len']
            
            # First segment: skip metadata
            if i == 0:
                start_pos = seg_offset + kv_data_start
                data_length = seg_length - kv_data_start
            else:
                # Other segments: all data is KV
                start_pos = seg_offset
                data_length = seg_length
            
            if data_length > 0:
                segment_data = bytes(shm_buffer[start_pos:start_pos + data_length])
                result.extend(segment_data)
        
        return bytes(result) if result else None
    
    @staticmethod
    def _validate_tensor(
        kv_tensor: torch.Tensor,
        expected_tokens: int,
        hidden_dim: int
    ) -> torch.Tensor:
        """
        Validate tensor size and pad if necessary.
        
        Args:
            kv_tensor: Raw KV tensor
            expected_tokens: Expected number of tokens
            hidden_dim: Hidden dimension size
            
        Returns:
            Validated/padded tensor
        """
        expected_elements = expected_tokens * 2 * hidden_dim
        actual_elements = len(kv_tensor)
        
        if actual_elements < expected_elements:
            # Pad with zeros (common with chunked storage)
            logger.debug(f"Padding tensor: {actual_elements} -> {expected_elements}")
            padded = torch.zeros(expected_elements, dtype=kv_tensor.dtype)
            padded[:actual_elements] = kv_tensor
            return padded
            
        elif actual_elements > expected_elements:
            # Truncate if too large
            logger.warning(f"Truncating tensor: {actual_elements} -> {expected_elements}")
            return kv_tensor[:expected_elements]
        
        return kv_tensor