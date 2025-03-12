"""
Relationship Detector Module for the Chunking Pipeline.
This module detects relationships between chunks (text, tables, charts).
"""

import logging
import re
from typing import Dict, Any, List, Optional, Union, Tuple
import copy

from ..utils.helpers import detect_references, calculate_similarity

# Configure logging
logger = logging.getLogger("chunking_cell.relationship_detector")

class RelationshipDetector:
    """
    Class for detecting relationships between chunks.
    Identifies references, related data, and adjacent content.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the relationship detector with configuration.
        
        Args:
            config: Configuration dictionary with detection settings
        """
        self.config = config
        
        # Get reference patterns
        self.table_reference_patterns = config.get("table_reference_patterns", 
                                                 ["Table", "table", "tbl"])
        self.chart_reference_patterns = config.get("chart_reference_patterns", 
                                                 ["Figure", "figure", "fig", "chart", "graph"])
        
        # Get other settings
        self.max_distance_for_adjacency = config.get("max_distance_for_adjacency", 3)
        self.min_confidence_threshold = config.get("min_confidence_threshold", 0.7)
        
        logger.info(f"Initialized RelationshipDetector with max_distance={self.max_distance_for_adjacency}, "
                   f"min_confidence={self.min_confidence_threshold}")
    
    def detect_relationships(
        self, 
        text_chunks: List[Dict[str, Any]], 
        table_chunks: List[Dict[str, Any]], 
        chart_chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Detect relationships between chunks.
        
        Args:
            text_chunks: List of text chunks
            table_chunks: List of table chunks
            chart_chunks: List of chart chunks
            
        Returns:
            List of relationships
        """
        logger.info("Starting relationship detection")
        
        relationships = []
        
        # Create lookup maps for tables and charts by page and index
        table_map = self._create_chunk_map(table_chunks)
        chart_map = self._create_chunk_map(chart_chunks)
        
        # Detect text-to-table references
        text_to_table_refs = self._detect_text_to_table_references(
            text_chunks, table_map
        )
        relationships.extend(text_to_table_refs)
        logger.info(f"Detected {len(text_to_table_refs)} text-to-table references")
        
        # Detect text-to-chart references
        text_to_chart_refs = self._detect_text_to_chart_references(
            text_chunks, chart_map
        )
        relationships.extend(text_to_chart_refs)
        logger.info(f"Detected {len(text_to_chart_refs)} text-to-chart references")
        
        # Detect table-to-chart relationships
        table_to_chart_refs = self._detect_table_to_chart_relationships(
            table_chunks, chart_chunks
        )
        relationships.extend(table_to_chart_refs)
        logger.info(f"Detected {len(table_to_chart_refs)} table-to-chart relationships")
        
        # Detect adjacent content
        adjacent_relationships = self._detect_adjacent_content(
            text_chunks, table_chunks, chart_chunks
        )
        relationships.extend(adjacent_relationships)
        logger.info(f"Detected {len(adjacent_relationships)} adjacent content relationships")
        
        logger.info(f"Completed relationship detection, found {len(relationships)} relationships total")
        return relationships
    
    def _create_chunk_map(self, chunks: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Create a lookup map for chunks by page and index.
        
        Args:
            chunks: List of chunks
            
        Returns:
            Map of chunks by page and index
        """
        chunk_map = {}
        
        for chunk in chunks:
            page_num = chunk.get("page_number", "unknown")
            
            # Get the index based on chunk type
            if chunk.get("type") == "table":
                index = chunk.get("table_index", 0)
            elif chunk.get("type") == "chart":
                index = chunk.get("chart_index", 0)
            else:
                continue
            
            # Create key
            key = f"{page_num}_{index}"
            chunk_map[key] = chunk
        
        return chunk_map
    
    def _detect_text_to_table_references(
        self, 
        text_chunks: List[Dict[str, Any]], 
        table_map: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Detect references from text chunks to tables.
        
        Args:
            text_chunks: List of text chunks
            table_map: Map of table chunks by page and index
            
        Returns:
            List of relationships
        """
        relationships = []
        
        for text_chunk in text_chunks:
            content = text_chunk.get("content", "")
            
            # Skip empty content
            if not content:
                continue
            
            # Detect table references
            references = detect_references(content, self.table_reference_patterns)
            
            # Process each reference
            for ref in references:
                # Try to match reference to a table
                target_chunk = self._match_reference_to_table(ref, text_chunk, table_map)
                
                if target_chunk:
                    relationship = {
                        "source_id": text_chunk["id"],
                        "target_id": target_chunk["id"],
                        "relationship_type": "reference",
                        "reference_text": ref,
                        "confidence": 0.9  # High confidence for explicit references
                    }
                    relationships.append(relationship)
        
        return relationships
    
    def _match_reference_to_table(
        self, 
        reference: str, 
        text_chunk: Dict[str, Any], 
        table_map: Dict[str, Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Match a reference to a table chunk.
        
        Args:
            reference: Reference text
            text_chunk: Text chunk containing the reference
            table_map: Map of table chunks by page and index
            
        Returns:
            Matching table chunk or None
        """
        # Try to extract a number from the reference
        num_match = re.search(r'\d+', reference)
        if not num_match:
            return None
        
        table_num = int(num_match.group(0))
        
        # Get the page number of the text chunk
        page_num = text_chunk.get("page_number", "unknown")
        
        # First, try to find a table on the same page
        key = f"{page_num}_{table_num - 1}"  # Subtract 1 because indices are 0-based
        if key in table_map:
            return table_map[key]
        
        # If not found, try nearby pages
        for offset in range(1, 3):
            # Try previous pages
            prev_page = f"{page_num - offset}_{table_num - 1}"
            if prev_page in table_map:
                return table_map[prev_page]
            
            # Try next pages
            next_page = f"{page_num + offset}_{table_num - 1}"
            if next_page in table_map:
                return table_map[next_page]
        
        # If still not found, try to match by table number only
        for key, table in table_map.items():
            if table.get("table_index", 0) == table_num - 1:
                return table
        
        return None
    
    def _detect_text_to_chart_references(
        self, 
        text_chunks: List[Dict[str, Any]], 
        chart_map: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Detect references from text chunks to charts.
        
        Args:
            text_chunks: List of text chunks
            chart_map: Map of chart chunks by page and index
            
        Returns:
            List of relationships
        """
        relationships = []
        
        for text_chunk in text_chunks:
            content = text_chunk.get("content", "")
            
            # Skip empty content
            if not content:
                continue
            
            # Detect chart references
            references = detect_references(content, self.chart_reference_patterns)
            
            # Process each reference
            for ref in references:
                # Try to match reference to a chart
                target_chunk = self._match_reference_to_chart(ref, text_chunk, chart_map)
                
                if target_chunk:
                    relationship = {
                        "source_id": text_chunk["id"],
                        "target_id": target_chunk["id"],
                        "relationship_type": "reference",
                        "reference_text": ref,
                        "confidence": 0.9  # High confidence for explicit references
                    }
                    relationships.append(relationship)
        
        return relationships
    
    def _match_reference_to_chart(
        self, 
        reference: str, 
        text_chunk: Dict[str, Any], 
        chart_map: Dict[str, Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Match a reference to a chart chunk.
        
        Args:
            reference: Reference text
            text_chunk: Text chunk containing the reference
            chart_map: Map of chart chunks by page and index
            
        Returns:
            Matching chart chunk or None
        """
        # Try to extract a number from the reference
        num_match = re.search(r'\d+', reference)
        if not num_match:
            return None
        
        chart_num = int(num_match.group(0))
        
        # Get the page number of the text chunk
        page_num = text_chunk.get("page_number", "unknown")
        
        # First, try to find a chart on the same page
        key = f"{page_num}_{chart_num - 1}"  # Subtract 1 because indices are 0-based
        if key in chart_map:
            return chart_map[key]
        
        # If not found, try nearby pages
        for offset in range(1, 3):
            # Try previous pages
            prev_page = f"{page_num - offset}_{chart_num - 1}"
            if prev_page in chart_map:
                return chart_map[prev_page]
            
            # Try next pages
            next_page = f"{page_num + offset}_{chart_num - 1}"
            if next_page in chart_map:
                return chart_map[next_page]
        
        # If still not found, try to match by chart number only
        for key, chart in chart_map.items():
            if chart.get("chart_index", 0) == chart_num - 1:
                return chart
        
        return None
    
    def _detect_table_to_chart_relationships(
        self, 
        table_chunks: List[Dict[str, Any]], 
        chart_chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Detect relationships between tables and charts based on content similarity.
        
        Args:
            table_chunks: List of table chunks
            chart_chunks: List of chart chunks
            
        Returns:
            List of relationships
        """
        relationships = []
        
        for table_chunk in table_chunks:
            table_title = table_chunk.get("title", "")
            table_summary = table_chunk.get("summary", {}).get("brief_summary", "")
            
            # Skip tables without title or summary
            if not table_title and not table_summary:
                continue
            
            for chart_chunk in chart_chunks:
                chart_description = chart_chunk.get("description", "")
                chart_summary = chart_chunk.get("summary", {}).get("brief_summary", "")
                
                # Skip charts without description or summary
                if not chart_description and not chart_summary:
                    continue
                
                # Calculate similarity between table and chart
                similarity = self._calculate_table_chart_similarity(
                    table_title, table_summary, chart_description, chart_summary
                )
                
                # If similarity is above threshold, create relationship
                if similarity >= self.min_confidence_threshold:
                    relationship = {
                        "source_id": table_chunk["id"],
                        "target_id": chart_chunk["id"],
                        "relationship_type": "related_data",
                        "confidence": similarity
                    }
                    relationships.append(relationship)
        
        return relationships
    
    def _calculate_table_chart_similarity(
        self, 
        table_title: str, 
        table_summary: str, 
        chart_description: str, 
        chart_summary: str
    ) -> float:
        """
        Calculate similarity between a table and a chart.
        
        Args:
            table_title: Table title
            table_summary: Table summary
            chart_description: Chart description
            chart_summary: Chart summary
            
        Returns:
            Similarity score (0-1)
        """
        # Combine table and chart text
        table_text = f"{table_title} {table_summary}".strip()
        chart_text = f"{chart_description} {chart_summary}".strip()
        
        # Calculate similarity
        similarity = calculate_similarity(table_text, chart_text)
        
        return similarity
    
    def _detect_adjacent_content(
        self, 
        text_chunks: List[Dict[str, Any]], 
        table_chunks: List[Dict[str, Any]], 
        chart_chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Detect adjacent content based on page numbers and section context.
        
        Args:
            text_chunks: List of text chunks
            table_chunks: List of table chunks
            chart_chunks: List of chart chunks
            
        Returns:
            List of relationships
        """
        relationships = []
        
        # Combine all chunks
        all_chunks = []
        all_chunks.extend(text_chunks)
        all_chunks.extend(table_chunks)
        all_chunks.extend(chart_chunks)
        
        # Sort chunks by page number and position
        # Convert page_number to int before sorting to avoid str/int comparison issues
        def get_page_number(chunk):
            page_num = chunk.get("page_number", 0)
            if isinstance(page_num, str):
                try:
                    return int(page_num)
                except ValueError:
                    return 0
            return page_num
            
        sorted_chunks = sorted(all_chunks, key=lambda x: (
            get_page_number(x), 
            x.get("window_position", {}).get("start", 0) if x.get("type") == "text" else 0
        ))
        
        # Detect adjacent content
        for i in range(len(sorted_chunks) - 1):
            chunk1 = sorted_chunks[i]
            chunk2 = sorted_chunks[i + 1]
            
            # Skip if chunks are not on the same page or nearby pages
            page1 = chunk1.get("page_number", 0)
            page2 = chunk2.get("page_number", 0)
            
            if isinstance(page1, str):
                try:
                    page1 = int(page1)
                except ValueError:
                    page1 = 0
            
            if isinstance(page2, str):
                try:
                    page2 = int(page2)
                except ValueError:
                    page2 = 0
            
            page_distance = abs(page1 - page2)
            
            if page_distance > self.max_distance_for_adjacency:
                continue
            
            # Check if chunks are in the same section
            section1 = chunk1.get("section", "")
            section2 = chunk2.get("section", "")
            
            same_section = section1 and section2 and section1 == section2
            
            # Calculate confidence based on proximity and section context
            confidence = 0.0
            
            if same_section:
                confidence += 0.5
            
            if page_distance == 0:
                confidence += 0.5
            elif page_distance == 1:
                confidence += 0.3
            else:
                confidence += 0.1
            
            # If confidence is above threshold, create relationship
            if confidence >= self.min_confidence_threshold:
                relationship = {
                    "source_id": chunk1["id"],
                    "target_id": chunk2["id"],
                    "relationship_type": "adjacent",
                    "confidence": confidence
                }
                relationships.append(relationship)
        
        return relationships
