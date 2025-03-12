"""
Chart Chunker Module for the Chunking Pipeline.
This module processes chart content from PDF extraction and creates discrete chunks.
"""

import logging
from typing import Dict, Any, List, Optional, Union
import copy
import os

from ..utils.helpers import generate_chunk_id

# Configure logging
logger = logging.getLogger("chunking_cell.chart_chunker")

class ChartChunker:
    """
    Class for chunking chart content from PDF extraction.
    Each chart is processed as a single chunk.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the chart chunker with configuration.
        
        Args:
            config: Configuration dictionary with chunking settings
        """
        self.config = config
        self.include_summary = config.get("include_summary", True)
        self.include_trends = config.get("include_trends", True)
        self.include_insights = config.get("include_insights", True)
        self.include_extracted_table = config.get("include_extracted_table", True)
        self.include_image_path = config.get("include_image_path", True)
        
        logger.info(f"Initialized ChartChunker with include_summary={self.include_summary}, "
                   f"include_trends={self.include_trends}")
    
    def process(self, charts_json: Dict[str, Any], file_id: str = None, company_name: str = None) -> List[Dict[str, Any]]:
        """
        Process the chart content and create chunks.
        
        Args:
            charts_json: Chart content from PDF extraction
            file_id: Optional identifier for the source file
            company_name: Optional company name to include in metadata
            
        Returns:
            List of chart chunks
        """
        logger.info("Starting chart chunking process")
        
        chunks = []
        
        # Process each chart
        if "processed_images" in charts_json:
            images = charts_json["processed_images"]
            
            # Filter to only include charts
            charts = [img for img in images if img.get("is_chart", False)]
            
            for chart_idx, chart in enumerate(charts):
                # Get page number
                page_num = chart.get("page", "unknown")
                
                # Create chunk for this chart
                chunk = self._process_chart(chart, chart_idx, page_num, file_id, company_name)
                chunks.append(chunk)
                
                logger.info(f"Created chunk for chart {chart_idx} on page {page_num}")
        
        logger.info(f"Completed chart chunking process, created {len(chunks)} chunks total")
        return chunks
    
    def _process_chart(
        self, 
        chart: Dict[str, Any], 
        chart_idx: int, 
        page_num: Union[int, str],
        file_id: str = None,
        company_name: str = None
    ) -> Dict[str, Any]:
        """
        Process a single chart and create a chunk.
        
        Args:
            chart: Chart data
            chart_idx: Index of the chart
            page_num: Page number
            file_id: Optional identifier for the source file
            company_name: Optional company name to include in metadata
            
        Returns:
            Chart chunk dictionary
        """
        # Generate unique ID
        chunk_id = generate_chunk_id("chart", page_num, chart_idx, file_id)
        
        # Create base chunk
        chunk = {
            "id": chunk_id,
            "type": "chart",
            "page_number": page_num,
            "chart_index": chart_idx
        }
        
        # Add file_id if provided
        if file_id:
            chunk["file_id"] = file_id
            
        # Add company_name if provided
        if company_name:
            chunk["company_name"] = company_name
        
        # Add chart metadata
        if "chart_type" in chart:
            chunk["chart_type"] = chart["chart_type"]
        
        if "description" in chart:
            chunk["description"] = chart["description"]
        
        if "axes_labels" in chart:
            chunk["axes_labels"] = copy.deepcopy(chart["axes_labels"])
        
        if "data_series" in chart:
            chunk["data_series"] = copy.deepcopy(chart["data_series"])
        
        # Add extracted table if available and requested
        if self.include_extracted_table and "extracted_table" in chart:
            chunk["extracted_table"] = copy.deepcopy(chart["extracted_table"])
        
        # Add image path if available and requested
        if self.include_image_path and "path" in chart:
            # Store relative path instead of absolute path
            abs_path = chart["path"]
            rel_path = os.path.basename(abs_path)
            chunk["image_path"] = rel_path
        
        # Add AI enhancements if available
        if "ai_enhancements" in chart:
            enhancements = chart["ai_enhancements"]
            
            # Add summary if available and requested
            if self.include_summary and "summary" in enhancements:
                chunk["summary"] = copy.deepcopy(enhancements["summary"])
            
            # Add trends if available and requested
            if self.include_trends and "trends" in enhancements:
                chunk["trends"] = copy.deepcopy(enhancements["trends"])
            
            # Add insights if available and requested
            if self.include_insights and "insights" in enhancements:
                chunk["insights"] = copy.deepcopy(enhancements["insights"])
            
            # Add keywords if available
            if "keywords" in enhancements:
                chunk["keywords"] = copy.deepcopy(enhancements["keywords"])
            
            # Copy any custom prompt results
            for key, value in enhancements.items():
                if key not in ["summary", "trends", "insights", "keywords"]:
                    logger.info(f"Found custom prompt result for chart: {key}")
                    chunk[key] = copy.deepcopy(value)
        
        return chunk
