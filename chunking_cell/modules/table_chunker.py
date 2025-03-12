"""
Table Chunker Module for the Chunking Pipeline.
This module processes table content from PDF extraction and creates discrete chunks.
"""

import logging
from typing import Dict, Any, List, Optional, Union
import copy

from ..utils.helpers import generate_chunk_id

# Configure logging
logger = logging.getLogger("chunking_cell.table_chunker")

class TableChunker:
    """
    Class for chunking table content from PDF extraction.
    Each table is processed as a single chunk.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the table chunker with configuration.
        
        Args:
            config: Configuration dictionary with chunking settings
        """
        self.config = config
        self.include_summary = config.get("include_summary", True)
        self.include_insights = config.get("include_insights", True)
        self.include_keywords = config.get("include_keywords", True)
        self.include_metadata = config.get("include_metadata", True)
        
        logger.info(f"Initialized TableChunker with include_summary={self.include_summary}, "
                   f"include_insights={self.include_insights}")
    
    def process(self, tables_json: Dict[str, Any], file_id: str = None, company_name: str = None) -> List[Dict[str, Any]]:
        """
        Process the table content and create chunks.
        
        Args:
            tables_json: Table content from PDF extraction
            file_id: Optional identifier for the source file
            company_name: Optional company name to include in metadata
            
        Returns:
            List of table chunks
        """
        logger.info("Starting table chunking process")
        
        chunks = []
        
        # Process each table
        if "tables" in tables_json:
            tables = tables_json["tables"]
            
            for table_idx, table in enumerate(tables):
                # Skip tables without data
                if not table.get("data") and not table.get("headers"):
                    continue
                
                # Get page number
                page_num = table.get("page", table.get("page_number", "unknown"))
                
                # Create chunk for this table
                chunk = self._process_table(table, table_idx, page_num, file_id, company_name)
                chunks.append(chunk)
                
                logger.info(f"Created chunk for table {table_idx} on page {page_num}")
        
        logger.info(f"Completed table chunking process, created {len(chunks)} chunks total")
        return chunks
    
    def _process_table(
        self, 
        table: Dict[str, Any], 
        table_idx: int, 
        page_num: Union[int, str],
        file_id: str = None,
        company_name: str = None
    ) -> Dict[str, Any]:
        """
        Process a single table and create a chunk.
        
        Args:
            table: Table data
            table_idx: Index of the table
            page_num: Page number
            file_id: Optional identifier for the source file
            company_name: Optional company name to include in metadata
            
        Returns:
            Table chunk dictionary
        """
        # Generate unique ID
        chunk_id = generate_chunk_id("table", page_num, table_idx, file_id)
        
        # Create base chunk
        chunk = {
            "id": chunk_id,
            "type": "table",
            "page_number": page_num,
            "table_index": table_idx
        }
        
        # Add file_id if provided
        if file_id:
            chunk["file_id"] = file_id
            
        # Add company_name if provided
        if company_name:
            chunk["company_name"] = company_name
        
        # Add table data
        if "headers" in table:
            chunk["headers"] = copy.deepcopy(table["headers"])
        
        if "data" in table:
            chunk["rows"] = copy.deepcopy(table["data"])
        
        if "data_types" in table:
            chunk["data_types"] = copy.deepcopy(table["data_types"])
        
        if "units" in table:
            chunk["units"] = copy.deepcopy(table["units"])
        
        # Add table metadata if available
        if self.include_metadata:
            if "title" in table:
                chunk["title"] = table["title"]
            
            if "shape" in table:
                chunk["shape"] = table["shape"]
            
            if "footnotes" in table:
                chunk["footnotes"] = table["footnotes"]
            
            if "table_type" in table:
                chunk["table_type"] = table["table_type"]
            
            if "purpose" in table:
                chunk["purpose"] = table["purpose"]
            
            if "source" in table:
                chunk["source"] = table["source"]
            
            if "notes" in table:
                chunk["notes"] = table["notes"]
        
        # Add AI enhancements if available
        if "ai_enhancements" in table:
            enhancements = table["ai_enhancements"]
            
            # Add summary if available and requested
            if self.include_summary and "summary" in enhancements:
                chunk["summary"] = copy.deepcopy(enhancements["summary"])
            
            # Add insights if available and requested
            if self.include_insights and "insights" in enhancements:
                chunk["insights"] = copy.deepcopy(enhancements["insights"])
            
            # Add keywords if available and requested
            if self.include_keywords and "keywords" in enhancements:
                chunk["keywords"] = copy.deepcopy(enhancements["keywords"])
            
            # Copy any custom prompt results
            for key, value in enhancements.items():
                if key not in ["summary", "insights", "keywords"]:
                    logger.info(f"Found custom prompt result for table: {key}")
                    chunk[key] = copy.deepcopy(value)
        
        return chunk
