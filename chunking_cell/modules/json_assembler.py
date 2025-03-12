"""
JSON Assembler Module for the Chunking Pipeline.
This module assembles the final unified JSON output from all components.
"""

import logging
from typing import Dict, Any, List, Optional, Union
import copy
from datetime import datetime
import time

# Configure logging
logger = logging.getLogger("chunking_cell.json_assembler")

class JSONAssembler:
    """
    Class for assembling the final unified JSON output from all components.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the JSON assembler with configuration.
        
        Args:
            config: Configuration dictionary with assembler settings
        """
        self.config = config
        logger.info("Initialized JSONAssembler")
    
    def assemble(
        self, 
        metadata: Dict[str, Any], 
        text_chunks: List[Dict[str, Any]], 
        table_chunks: List[Dict[str, Any]], 
        chart_chunks: List[Dict[str, Any]], 
        relationships: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Assemble the final unified JSON output from all components.
        
        Args:
            metadata: Document metadata
            text_chunks: List of text chunks
            table_chunks: List of table chunks
            chart_chunks: List of chart chunks
            relationships: List of relationships between chunks
            
        Returns:
            Unified JSON output
        """
        logger.info("Starting JSON assembly")
        start_time = time.time()
        
        # Create the base output structure
        output = {
            "metadata": copy.deepcopy(metadata),
            "chunks": [],
            "relationships": copy.deepcopy(relationships),
            "processing_info": {
                "timestamp": datetime.now().isoformat(),
                "version": "1.0.0"
            }
        }
        
        # Combine all chunks
        all_chunks = []
        all_chunks.extend(copy.deepcopy(text_chunks))
        all_chunks.extend(copy.deepcopy(table_chunks))
        all_chunks.extend(copy.deepcopy(chart_chunks))
        
        # Sort chunks by page number and position
        sorted_chunks = sorted(all_chunks, key=lambda x: (
            self._get_page_number(x), 
            self._get_position(x)
        ))
        
        output["chunks"] = sorted_chunks
        
        # Add processing statistics
        processing_time = time.time() - start_time
        output["processing_info"]["processing_time"] = round(processing_time, 3)
        output["processing_info"]["chunk_count"] = len(sorted_chunks)
        output["processing_info"]["relationship_count"] = len(relationships)
        output["processing_info"]["chunk_type_counts"] = {
            "text": len(text_chunks),
            "table": len(table_chunks),
            "chart": len(chart_chunks)
        }
        
        logger.info(f"Completed JSON assembly with {len(sorted_chunks)} chunks and "
                   f"{len(relationships)} relationships")
        return output
    
    def _get_page_number(self, chunk: Dict[str, Any]) -> int:
        """
        Get the page number from a chunk, converting to int if necessary.
        
        Args:
            chunk: Chunk dictionary
            
        Returns:
            Page number as int
        """
        page_num = chunk.get("page_number", 0)
        
        if isinstance(page_num, str):
            try:
                return int(page_num)
            except ValueError:
                return 0
        
        return page_num
    
    def _get_position(self, chunk: Dict[str, Any]) -> int:
        """
        Get the position of a chunk on a page.
        For text chunks, use window_position.start.
        For table and chart chunks, use index.
        
        Args:
            chunk: Chunk dictionary
            
        Returns:
            Position as int
        """
        chunk_type = chunk.get("type", "")
        
        if chunk_type == "text":
            return chunk.get("window_position", {}).get("start", 0)
        elif chunk_type == "table":
            return chunk.get("table_index", 0) * 10000  # Multiply to ensure tables come after text
        elif chunk_type == "chart":
            return chunk.get("chart_index", 0) * 10000  # Multiply to ensure charts come after text
        
        return 0
