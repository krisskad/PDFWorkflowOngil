"""
Helper utilities for the chunking pipeline.
"""

import json
import os
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import re

# Configure logging
logger = logging.getLogger("chunking_cell.utils")

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {str(e)}")
        raise

def read_json(file_path: str) -> Dict[str, Any]:
    """
    Read JSON data from a file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Parsed JSON data
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Read JSON data from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Failed to read JSON data: {str(e)}")
        raise

def write_json(data: Dict[str, Any], file_path: str) -> None:
    """
    Write JSON data to a file.
    
    Args:
        data: Data to write
        file_path: Path to the output file
    """
    try:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Wrote JSON data to {file_path}")
    except Exception as e:
        logger.error(f"Failed to write JSON data: {str(e)}")
        raise

def generate_chunk_id(chunk_type: str, page_number: Union[int, str], index: int, file_id: str = None) -> str:
    """
    Generate a unique ID for a chunk.
    
    Args:
        chunk_type: Type of chunk (text, table, chart)
        page_number: Page number
        index: Index of the chunk on the page
        file_id: Optional identifier for the source file
        
    Returns:
        Unique chunk ID
    """
    if file_id:
        return f"{file_id}_{chunk_type}_p{page_number}_{index}"
    else:
        return f"{chunk_type}_p{page_number}_{index}"

def extract_page_number(text: str) -> Optional[int]:
    """
    Extract page number from text.
    
    Args:
        text: Text to extract page number from
        
    Returns:
        Page number if found, None otherwise
    """
    # Look for page markers like "Page X" or "--- Page X ---"
    page_match = re.search(r'---\s*Page\s+(\d+)\s*---', text)
    if page_match:
        return int(page_match.group(1))
    
    # Try other patterns
    page_match = re.search(r'Page\s+(\d+)', text)
    if page_match:
        return int(page_match.group(1))
    
    return None

def filter_references(text: str, reference_patterns: List[str]) -> str:
    """
    Filter out references to tables or charts from text.
    
    Args:
        text: Text to filter
        reference_patterns: Patterns to filter out
        
    Returns:
        Filtered text
    """
    # Create a regex pattern that matches any of the reference patterns
    # followed by a number or identifier
    pattern = r'(' + '|'.join(reference_patterns) + r')\s*[\d\w\-\.]+\b'
    
    # Replace matches with empty string
    filtered_text = re.sub(pattern, '', text)
    
    return filtered_text

def detect_references(text: str, reference_patterns: List[str]) -> List[str]:
    """
    Detect references to tables or charts in text.
    
    Args:
        text: Text to search for references
        reference_patterns: Patterns to look for
        
    Returns:
        List of detected references
    """
    # Create a regex pattern that matches any of the reference patterns
    # followed by a number or identifier
    pattern = r'(' + '|'.join(reference_patterns) + r')\s*([\d\w\-\.]+)\b'
    
    # Find all matches
    matches = re.findall(pattern, text)
    
    # Extract the identifiers
    references = [match[1] for match in matches]
    
    return references

def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two text strings.
    Simple implementation using word overlap.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score (0-1)
    """
    # Convert to lowercase and split into words
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    # Calculate Jaccard similarity
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    if union == 0:
        return 0.0
    
    return intersection / union
