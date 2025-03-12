"""
Document Processor Module for the Chunking Pipeline.
This module extracts document-level metadata from PDF extraction outputs.
"""

import logging
from typing import Dict, Any, List, Optional, Union
import copy
import os
from datetime import datetime

# Configure logging
logger = logging.getLogger("chunking_cell.document_processor")

class DocumentProcessor:
    """
    Class for extracting document-level metadata from PDF extraction outputs.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the document processor with configuration.
        
        Args:
            config: Configuration dictionary with processing settings
        """
        self.config = config
        self.include_summary = config.get("include_summary", True)
        self.include_topics = config.get("include_topics", True)
        self.include_entities = config.get("include_entities", True)
        self.include_structure = config.get("include_structure", True)
        self.include_key_information = config.get("include_key_information", True)
        
        logger.info(f"Initialized DocumentProcessor with include_summary={self.include_summary}, "
                   f"include_topics={self.include_topics}")
    
    def extract_metadata(
        self, 
        text_json: Dict[str, Any], 
        tables_json: Dict[str, Any], 
        charts_json: Dict[str, Any],
        document_id: str = None
    ) -> Dict[str, Any]:
        """
        Extract document-level metadata from PDF extraction outputs.
        
        Args:
            text_json: Text content from PDF extraction
            tables_json: Table content from PDF extraction
            charts_json: Chart content from PDF extraction
            document_id: Unique identifier for the document
            
        Returns:
            Document metadata dictionary
        """
        logger.info("Starting document metadata extraction")
        
        metadata = {}
        
        # Add document ID if provided
        if document_id:
            metadata["document_id"] = document_id
        
        # Extract file metadata
        file_metadata = self._extract_file_metadata(text_json, tables_json, charts_json)
        metadata["file_info"] = file_metadata
        
        # Extract AI enhancements from text JSON
        ai_enhancements = text_json.get("ai_enhancements", {})
        
        # Extract document type
        if "document_type" in ai_enhancements:
            metadata["document_type"] = copy.deepcopy(ai_enhancements["document_type"])
        
        # Extract summary if available and requested
        if self.include_summary and "summary" in ai_enhancements:
            metadata["summary"] = copy.deepcopy(ai_enhancements["summary"])
        
        # Extract topics if available and requested
        if self.include_topics and "topics" in ai_enhancements:
            metadata["topics"] = copy.deepcopy(ai_enhancements["topics"])
        
        # Extract entities if available and requested
        if self.include_entities and "entities" in ai_enhancements:
            metadata["entities"] = copy.deepcopy(ai_enhancements["entities"])
        
        # Extract document structure if available and requested
        if self.include_structure and "document_structure" in ai_enhancements:
            metadata["document_structure"] = copy.deepcopy(ai_enhancements["document_structure"])
        
        # Extract key information if available and requested
        if self.include_key_information and "key_information" in ai_enhancements:
            metadata["key_information"] = copy.deepcopy(ai_enhancements["key_information"])
        
        # Copy any custom prompt results
        for key, value in ai_enhancements.items():
            if key not in ["document_type", "summary", "topics", "entities",
                          "document_structure", "key_information", "page_level_analysis"]:
                logger.info(f"Found custom prompt result: {key}")
                metadata[key] = copy.deepcopy(value)
        
        # Add processing metadata
        metadata["processing_info"] = {
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0"
        }
        
        logger.info("Completed document metadata extraction")
        return metadata
    
    def _extract_file_metadata(
        self, 
        text_json: Dict[str, Any], 
        tables_json: Dict[str, Any], 
        charts_json: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract file metadata from PDF extraction outputs.
        
        Args:
            text_json: Text content from PDF extraction
            tables_json: Table content from PDF extraction
            charts_json: Chart content from PDF extraction
            
        Returns:
            File metadata dictionary
        """
        file_metadata = {}
        
        # Try to extract filename from text JSON
        if "original_content" in text_json and "file_path" in text_json["original_content"]:
            file_path = text_json["original_content"]["file_path"]
            file_metadata["name"] = os.path.basename(file_path)
            file_metadata["path"] = file_path
        
        # Try to extract creation date
        if "original_content" in text_json and "metadata" in text_json["original_content"]:
            pdf_metadata = text_json["original_content"]["metadata"]
            
            if "creation_date" in pdf_metadata:
                file_metadata["creation_date"] = pdf_metadata["creation_date"]
            
            if "author" in pdf_metadata:
                file_metadata["author"] = pdf_metadata["author"]
            
            if "title" in pdf_metadata:
                file_metadata["title"] = pdf_metadata["title"]
        
        # Add document statistics
        stats = self._calculate_document_statistics(text_json, tables_json, charts_json)
        file_metadata["statistics"] = stats
        
        return file_metadata
    
    def _calculate_document_statistics(
        self, 
        text_json: Dict[str, Any], 
        tables_json: Dict[str, Any], 
        charts_json: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate document statistics from PDF extraction outputs.
        
        Args:
            text_json: Text content from PDF extraction
            tables_json: Table content from PDF extraction
            charts_json: Chart content from PDF extraction
            
        Returns:
            Document statistics dictionary
        """
        stats = {}
        
        # Count pages
        if "original_content" in text_json and "text" in text_json["original_content"]:
            if "pages" in text_json["original_content"]["text"]:
                stats["page_count"] = len(text_json["original_content"]["text"]["pages"])
        
        # Count tables
        if "tables" in tables_json:
            stats["table_count"] = len(tables_json["tables"])
        
        # Count charts
        if "processed_images" in charts_json:
            # Count only images that are charts
            chart_count = sum(1 for img in charts_json["processed_images"] 
                             if img.get("is_chart", False))
            stats["chart_count"] = chart_count
        
        return stats
