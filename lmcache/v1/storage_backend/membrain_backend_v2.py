"""
Simplified Membrain storage backend.
"""

import asyncio
import logging
from typing import Optional
from concurrent.futures import Future

from lmcache.v1.storage_backend.remote_backend import RemoteBackend
from lmcache.v1.storage_backend.connector import parse_remote_url
from lmcache.v1.storage_backend.connector import InstrumentedRemoteConnector
from lmcache.v1.membrain_connector_v2 import SimplifiedMembrainConnector
from lmcache.v1.memory_management import MemoryObj
from lmcache.utils import CacheEngineKey

logger = logging.getLogger(__name__)


class SimplifiedMembrainBackend(RemoteBackend):
    """
    Simplified Membrain backend using the streamlined connector.
    """
    
    def _init_connection(self):
        """Initialize Membrain connection."""
        if self.connection is not None:
            return
            
        try:
            assert self.config.remote_url is not None
            
            # Parse URL to extract endpoint and namespace
            parsed_url = parse_remote_url(self.config.remote_url)
            
            if len(parsed_url.hosts) != 1:
                raise ValueError(
                    f"Membrain only supports single host, got: {self.config.remote_url}"
                )
            
            host = parsed_url.hosts[0]
            port = parsed_url.ports[0]
            endpoint = f"http://{host}:{port}"
            namespace = parsed_url.query_params[0].get("namespace", "lmcache")
            
            # Create simplified connector
            connector = SimplifiedMembrainConnector(
                endpoint, namespace, self.loop, self.local_cpu_backend
            )
            
            # Wrap with instrumentation
            self.connection = InstrumentedRemoteConnector(connector)
            
            logger.info(f"Initialized Membrain connection: {endpoint}/{namespace}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Membrain connection: {e}")
            self.connection = None
    
    def get_non_blocking(self, key: CacheEngineKey) -> Optional[Future]:
        """Non-blocking get operation."""
        if self.connection is None:
            self._init_connection()
            if self.connection is None:
                return None
        
        return asyncio.run_coroutine_threadsafe(
            self._get_async(key), self.loop
        )
    
    async def _get_async(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """Async get operation with deserialization."""
        try:
            memory_obj = await self.connection.get(key)
            if memory_obj is None:
                return None
            
            # For Membrain, we return the lease object directly
            # No deserialization needed for zero-copy path
            return memory_obj
            
        except Exception as e:
            logger.error(f"Error in get operation: {e}")
            self.connection = None
            return None
    
    def unpin(self, key: CacheEngineKey) -> bool:
        """Unpin operation (no-op for Membrain)."""
        return True