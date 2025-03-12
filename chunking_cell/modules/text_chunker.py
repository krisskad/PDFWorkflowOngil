"""
Text Chunker Module for the Chunking Pipeline.
This module processes text content from PDF extraction and creates discrete chunks.
"""

import logging
import re
from typing import Dict, Any, List, Optional, Union
import copy

from ..utils.helpers import generate_chunk_id, extract_page_number, filter_references

# Configure logging
logger = logging.getLogger("chunking_cell.text_chunker")

class TextChunker:
    """
    Class for chunking text content from PDF extraction.
    Implements page-level chunking with sliding window.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the text chunker with configuration.
        
        Args:
            config: Configuration dictionary with chunking settings
        """
        self.config = config
        self.chunk_size = config.get("chunk_size", 1000)
        self.chunk_overlap = config.get("chunk_overlap", 200)
        self.min_chunk_size = config.get("min_chunk_size", 100)
        self.include_page_info = config.get("include_page_info", True)
        self.include_section_info = config.get("include_section_info", True)
        
        # Reference patterns for filtering
        self.table_reference_patterns = config.get("relationship_detection", {}).get(
            "table_reference_patterns", ["Table", "table", "tbl"]
        )
        self.chart_reference_patterns = config.get("relationship_detection", {}).get(
            "chart_reference_patterns", ["Figure", "figure", "fig", "chart", "graph"]
        )
        
        logger.info(f"Initialized TextChunker with chunk_size={self.chunk_size}, "
                   f"chunk_overlap={self.chunk_overlap}")
    
    def process(self, text_json: Dict[str, Any], file_id: str = None, company_name: str = None) -> List[Dict[str, Any]]:
        """
        Process the text content and create chunks.
        
        Args:
            text_json: Text content from PDF extraction
            file_id: Optional identifier for the source file
            company_name: Optional company name to include in metadata
            
        Returns:
            List of text chunks
        """
        logger.info("Starting text chunking process")
        
        # Extract the original content and AI enhancements
        original_content = text_json.get("original_content", {})
        ai_enhancements = text_json.get("ai_enhancements", {})
        
        # Get document-level metadata
        document_type = ai_enhancements.get("document_type", {}).get("type", "unknown")
        document_structure = ai_enhancements.get("document_structure", [])
        
        # Process each page
        chunks = []
        
        if "text" in original_content and "pages" in original_content["text"]:
            pages = original_content["text"]["pages"]
            
            # Get page-level analysis if available
            page_analyses = ai_enhancements.get("page_level_analysis", [])
            page_analysis_map = {
                analysis.get("page_number"): analysis 
                for analysis in page_analyses
            }
            
            for page_idx, page in enumerate(pages):
                if "content" in page and page["content"].strip():
                    page_num = page.get("page_number", page_idx + 1)
                    page_content = page["content"]
                    
                    # Get page analysis for this page
                    page_analysis = page_analysis_map.get(page_num, {})
                    
                    # Get section context for this page
                    section_context = self._get_section_context(
                        page_num, document_structure
                    )
                    
                    # Create chunks for this page
                    page_chunks = self._chunk_page_text(
                        page_content, 
                        page_num, 
                        page_analysis, 
                        section_context,
                        document_type,
                        file_id,
                        company_name
                    )
                    
                    chunks.extend(page_chunks)
                    logger.info(f"Created {len(page_chunks)} chunks for page {page_num}")
        
        logger.info(f"Completed text chunking process, created {len(chunks)} chunks total")
        return chunks
    
    def _chunk_page_text(
        self, 
        page_content: str, 
        page_num: Union[int, str], 
        page_analysis: Dict[str, Any],
        section_context: Dict[str, Any],
        document_type: str,
        file_id: str = None,
        company_name: str = None
    ) -> List[Dict[str, Any]]:
        """
        Create chunks for a single page using sliding window.
        
        Args:
            page_content: Text content of the page
            page_num: Page number
            page_analysis: AI enhancements for this page
            section_context: Section context for this page
            document_type: Document type
            file_id: Optional identifier for the source file
            company_name: Optional company name to include in metadata
            
        Returns:
            List of chunks for this page
        """
        chunks = []
        
        # Clean the page content (remove excessive whitespace)
        page_content = re.sub(r'\s+', ' ', page_content).strip()
        
        # If page content is too short, create a single chunk
        if len(page_content) <= self.min_chunk_size:
            if len(page_content) > 0:
                chunk = self._create_chunk(
                    page_content, 
                    page_num, 
                    0, 
                    page_analysis,
                    section_context,
                    0,
                    len(page_content),
                    file_id,
                    company_name
                )
                chunks.append(chunk)
            return chunks
        
        # Use sliding window to create chunks
        start = 0
        chunk_index = 0
        
        while start < len(page_content):
            # Calculate end position
            end = min(start + self.chunk_size, len(page_content))
            
            # If we're near the end, just include the rest
            if len(page_content) - end < self.min_chunk_size:
                end = len(page_content)
            
            # Extract chunk text
            chunk_text = page_content[start:end]
            
            # Create chunk
            chunk = self._create_chunk(
                chunk_text, 
                page_num, 
                chunk_index, 
                page_analysis,
                section_context,
                start,
                end,
                file_id,
                company_name
            )
            
            chunks.append(chunk)
            chunk_index += 1
            
            # Move start position for next chunk
            if end == len(page_content):
                break
            
            start = end - self.chunk_overlap
        
        return chunks
    
    def _create_chunk(
        self, 
        text: str, 
        page_num: Union[int, str], 
        chunk_index: int, 
        page_analysis: Dict[str, Any],
        section_context: Dict[str, Any],
        start_pos: int,
        end_pos: int,
        file_id: str = None,
        company_name: str = None
    ) -> Dict[str, Any]:
        """
        Create a single text chunk with metadata.
        
        Args:
            text: Chunk text content
            page_num: Page number
            chunk_index: Index of the chunk on the page
            page_analysis: AI enhancements for this page
            section_context: Section context for this page
            start_pos: Start position in the page text
            end_pos: End position in the page text
            file_id: Optional identifier for the source file
            company_name: Optional company name to include in metadata
            
        Returns:
            Chunk dictionary
        """
        # Generate unique ID
        chunk_id = generate_chunk_id("text", page_num, chunk_index, file_id)
        
        # Filter out references to tables and charts
        filtered_text = text
        if self.table_reference_patterns:
            filtered_text = filter_references(filtered_text, self.table_reference_patterns)
        if self.chart_reference_patterns:
            filtered_text = filter_references(filtered_text, self.chart_reference_patterns)
        
        # Create chunk
        chunk = {
            "id": chunk_id,
            "type": "text",
            "page_number": page_num,
            "content": filtered_text,
            "window_position": {
                "start": start_pos,
                "end": end_pos
            }
        }
        
        # Add file_id if provided
        if file_id:
            chunk["file_id"] = file_id
            
        # Add company_name if provided
        if company_name:
            chunk["company_name"] = company_name
        
        # Add page title if available
        if self.include_page_info and "title" in page_analysis:
            chunk["page_title"] = page_analysis["title"]
        
        # Add section context if available
        if self.include_section_info and section_context:
            chunk["section"] = section_context.get("title", "")
            chunk["section_level"] = section_context.get("level", 0)
            if "content_summary" in section_context:
                chunk["section_summary"] = section_context["content_summary"]
        
        # Add topics if available
        if "topics" in page_analysis:
            # Make a deep copy to avoid modifying the original
            topics = copy.deepcopy(page_analysis["topics"])
            
            # Handle different topic structures
            # Some formats have keywords as a dictionary mapped to main topics
            if "main_topics" in topics and "keywords" in topics and isinstance(topics["keywords"], dict):
                # Keep the structure as is, it's already well-organized
                pass
            # Some formats have keywords as a simple array
            elif "main_topics" in topics and "keywords" in topics and isinstance(topics["keywords"], list):
                # Keep the structure as is
                pass
            # If topics is just a list, convert to expected format
            elif isinstance(topics, list):
                topics = {
                    "main_topics": topics,
                    "keywords": []
                }
                
            chunk["topics"] = topics
        
        # Add entities if available
        if "entities" in page_analysis:
            # Make a deep copy to avoid modifying the original
            entities = copy.deepcopy(page_analysis["entities"])
            
            # Handle different entity structures
            # Check if entities is a dictionary with categories as keys
            if isinstance(entities, dict) and any(isinstance(entities.get(key), list) for key in entities):
                # This is the expected format with categories (People, Organizations, etc.)
                # Filter out empty categories
                filtered_entities = {k: v for k, v in entities.items() if v}
                chunk["entities"] = filtered_entities
            else:
                # If it's a flat list or other format, keep as is
                chunk["entities"] = entities
        
        return chunk
    
    def _get_section_context(
        self, 
        page_num: Union[int, str], 
        document_structure: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Get the section context for a page.
        
        Args:
            page_num: Page number
            document_structure: Document structure from AI enhancements
            
        Returns:
            Section context dictionary
        """
        if not document_structure:
            return {}
        
        # Convert page_num to int if it's a string
        if isinstance(page_num, str):
            try:
                page_num = int(page_num)
            except ValueError:
                return {}
        
        # Find the section that contains this page
        for section in document_structure:
            page_range = section.get("page_range", "")
            
            # Parse page range (e.g., "1-5" or "7")
            if "-" in page_range:
                try:
                    start_page, end_page = map(int, page_range.split("-"))
                    if start_page <= page_num <= end_page:
                        return section
                except ValueError:
                    continue
            else:
                try:
                    if int(page_range) == page_num:
                        return section
                except ValueError:
                    continue
            
            # Recursively check subsections
            if "subsections" in section:
                subsection = self._get_section_context(page_num, section["subsections"])
                if subsection:
                    return subsection
        
        return {}
