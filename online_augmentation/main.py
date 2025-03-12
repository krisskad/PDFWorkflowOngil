"""
Main entry point for the Online Augmentation Pipeline.
This script processes input files based on configuration to extract specific entities.
It can also perform online searches for the extracted entities using the Perplexity LLM.
"""

import os
import sys
import logging
import argparse
import json
import glob
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import time
from dotenv import load_dotenv

# Import online_search functionality
from online_augmentation.online_search import (
    process_file as process_search,
    extract_requested_entities,
    format_entities_for_search,
    perform_perplexity_search
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("online_augmentation.main")

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration as a dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {e}")
        raise

def read_json(file_path: str) -> Dict[str, Any]:
    """
    Read a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        JSON content as a dictionary
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Error reading JSON file {file_path}: {e}")
        raise

def write_json(data: Dict[str, Any], file_path: str) -> None:
    """
    Write data to a JSON file.
    
    Args:
        data: Data to write
        file_path: Path to the output file
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Error writing JSON file {file_path}: {e}")
        raise

def extract_entities(input_data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract entities from input data based on configuration.
    
    Args:
        input_data: Input data as a dictionary
        config: Configuration as a dictionary
        
    Returns:
        Extracted entities as a dictionary
    """
    result = {}
    
    # Extract metadata if enabled
    if config.get("metadata_extraction", {}).get("enabled", False):
        metadata_fields = config.get("metadata_extraction", {}).get("fields", [])
        metadata = {}
        for field in metadata_fields:
            if field in input_data.get("metadata", {}):
                metadata[field] = input_data["metadata"][field]
        
        if metadata:
            result["metadata"] = metadata
    
    # Extract entities if enabled
    if config.get("entity_extraction", {}).get("enabled", False):
        entity_types = config.get("entity_extraction", {}).get("entity_types", [])
        confidence_threshold = config.get("entity_extraction", {}).get("confidence_threshold", 0.0)
        include_context = config.get("entity_extraction", {}).get("include_context", False)
        max_entities_per_type = config.get("entity_extraction", {}).get("max_entities_per_type", 100)
        
        entities = {}
        
        # Extract entities from metadata
        for entity_type in entity_types:
            # Check if entity_type exists directly in metadata
            if entity_type in input_data.get("metadata", {}):
                entities[entity_type] = input_data["metadata"][entity_type]
            # Check if entity_type exists in metadata.entities
            elif entity_type in input_data.get("metadata", {}).get("entities", {}):
                entities[entity_type] = input_data["metadata"]["entities"].get(entity_type, [])
                
                # Filter by confidence threshold if it's a list of entities
                if isinstance(entities[entity_type], list) and confidence_threshold > 0:
                    entities[entity_type] = [
                        entity for entity in entities[entity_type]
                        if entity.get("confidence", 0) >= confidence_threshold
                    ]
                
                # Limit number of entities if it's a list
                if isinstance(entities[entity_type], list) and max_entities_per_type > 0:
                    entities[entity_type] = entities[entity_type][:max_entities_per_type]
                
                # Remove context if not needed and it's a list of entities
                if not include_context and isinstance(entities[entity_type], list):
                    for entity in entities[entity_type]:
                        if isinstance(entity, dict) and "context" in entity:
                            del entity["context"]
        
        # Extract entities from chunks
        for chunk in input_data.get("chunks", []):
            for entity_type in entity_types:
                # Check if entity_type exists directly in chunk
                if entity_type in chunk:
                    if entity_type not in entities:
                        entities[entity_type] = []
                    
                    # Handle different data structures
                    chunk_data = chunk[entity_type]
                    
                    # If it's a dictionary with a list of items
                    if isinstance(chunk_data, dict) and any(isinstance(chunk_data.get(k), list) for k in chunk_data):
                        for key, items in chunk_data.items():
                            if isinstance(items, list):
                                # Add chunk ID to items
                                for item in items:
                                    if isinstance(item, dict):
                                        item["chunk_id"] = chunk["id"]
                                
                                # Add to entities
                                if isinstance(entities[entity_type], list):
                                    entities[entity_type].extend(items)
                                else:
                                    # Initialize as a list if it wasn't already
                                    entities[entity_type] = items
                    # If it's a list
                    elif isinstance(chunk_data, list):
                        # Add chunk ID to items
                        for item in chunk_data:
                            if isinstance(item, dict):
                                item["chunk_id"] = chunk["id"]
                        
                        # Add to entities
                        if isinstance(entities[entity_type], list):
                            entities[entity_type].extend(chunk_data)
                        else:
                            # Initialize as a list if it wasn't already
                            entities[entity_type] = chunk_data
                
                # Check if entity_type exists in chunk.entities
                if entity_type in chunk.get("entities", {}):
                    if entity_type not in entities:
                        entities[entity_type] = []
                    
                    # Add chunk entities to the list
                    chunk_entities = chunk["entities"].get(entity_type, [])
                    
                    # Filter by confidence threshold
                    if confidence_threshold > 0:
                        chunk_entities = [
                            entity for entity in chunk_entities
                            if entity.get("confidence", 0) >= confidence_threshold
                        ]
                    
                    # Add chunk ID to entities
                    for entity in chunk_entities:
                        if isinstance(entity, dict):
                            entity["chunk_id"] = chunk["id"]
                    
                    entities[entity_type].extend(chunk_entities)
        
        # Deduplicate entities
        for entity_type in entities:
            # Only deduplicate if it's a list
            if isinstance(entities[entity_type], list):
                # Create a set to track duplicates
                seen_items = set()
                deduplicated_entities = []
                
                for entity in entities[entity_type]:
                    # Skip if not a dictionary
                    if not isinstance(entity, dict):
                        deduplicated_entities.append(entity)
                        continue
                    
                    # Try to find a unique identifier for deduplication
                    identifier = None
                    
                    # First try using 'text' field if it exists
                    if "text" in entity:
                        identifier = f"text:{entity['text']}"
                    # If no text field, try using 'section_name' (for Road_Sections)
                    elif "section_name" in entity:
                        identifier = f"section_name:{entity['section_name']}"
                    # If no section_name, try using 'complete_name' (for Road_Sections)
                    elif "complete_name" in entity:
                        identifier = f"complete_name:{entity['complete_name']}"
                    # If no identifiable field, use the whole entity as a string
                    else:
                        # Convert entity to a string representation for deduplication
                        # Sort keys to ensure consistent string representation
                        identifier = str(sorted(entity.items()))
                    
                    # Check if we've seen this identifier before
                    if identifier and identifier not in seen_items:
                        seen_items.add(identifier)
                        deduplicated_entities.append(entity)
                
                # Update entities with deduplicated list
                entities[entity_type] = deduplicated_entities
                
                # Limit number of entities after deduplication
                if max_entities_per_type > 0:
                    entities[entity_type] = entities[entity_type][:max_entities_per_type]
        
        if entities:
            result["entities"] = entities
    
    return result

def process_file(
    input_file_path: str,
    output_file_path: str = None,
    config_path: str = "online_augmentation/config/augmentation_config.json",
    run_online_search: bool = False,
    api_key: str = None
) -> Dict[str, Any]:
    """
    Process a file through the online augmentation pipeline.
    
    Args:
        input_file_path: Path to the input file
        output_file_path: Path to save the output file (optional)
        config_path: Path to the configuration file
        run_online_search: Whether to run online search after extraction
        api_key: OpenRouter API key for online search
        
    Returns:
        Processed data as a dictionary
    """
    start_time = time.time()
    logger.info(f"Starting online augmentation pipeline for file: {input_file_path}")
    
    # Load configuration
    config = load_config(config_path)
    logger.info(f"Loaded configuration from {config_path}")
    
    # If output_file_path is not provided, construct it from the input path
    if not output_file_path:
        input_path = Path(input_file_path)
        output_dir = config.get("output", {}).get("output_dir", "./data/output/augmented/")
        os.makedirs(output_dir, exist_ok=True)
        suffix = config.get("output", {}).get("file_name_suffix", "_augmented")
        output_file_path = os.path.join(output_dir, f"{input_path.stem}{suffix}{input_path.suffix}")
    
    logger.info(f"Input file path: {input_file_path}")
    logger.info(f"Output file path: {output_file_path}")
    
    # Read input file
    input_data = read_json(input_file_path)
    logger.info(f"Loaded input file: {len(input_data) if isinstance(input_data, list) else 'dictionary'} items")
    
    # Extract entities based on configuration
    output_data = extract_entities(input_data, config)
    
    # Add processing metadata
    output_data["processing_info"] = {
        "timestamp": time.time(),
        "input_file": input_file_path,
        "config_file": config_path
    }
    
    # Write output to file
    write_json(output_data, output_file_path)
    logger.info(f"Wrote output to {output_file_path}")
    
    # Log processing time for extraction
    extraction_time = time.time() - start_time
    logger.info(f"Completed entity extraction in {extraction_time:.2f} seconds")
    
    # Run online search if enabled
    if run_online_search:
        search_start_time = time.time()
        logger.info("Starting online search...")
        
        # Check if API key is provided
        if not api_key:
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                logger.error("No API key provided for online search. Please provide an API key via --api-key or set OPENROUTER_API_KEY in .env")
                return output_data
        
        # Get online search configuration
        online_search_config = config.get("online_search", {})
        entity_types = online_search_config.get("entity_types", ["Organizations", "People", "Locations"])
        max_entities = online_search_config.get("max_entities_per_type", 5)
        llm_model = online_search_config.get("llm_model", "perplexity/sonar-reasoning-pro")
        site_url = online_search_config.get("site_url", "https://example.com")
        site_name = online_search_config.get("site_name", "PDF Workflow")
        
        # Generate search output path
        search_output_path = output_file_path.replace("_augmented.json", "_searched.json")
        
        # Run online search
        search_result = process_search(
            input_file_path=output_file_path,
            entity_types=entity_types,
            api_key=api_key,
            output_file_path=search_output_path,
            max_entities_per_type=max_entities,
            llm_model=llm_model,
            site_url=site_url,
            site_name=site_name
        )
        
        # Log processing time for search
        search_time = time.time() - search_start_time
        logger.info(f"Completed online search in {search_time:.2f} seconds")
        
        # Return the search result
        return search_result
    
    # Log total processing time
    total_time = time.time() - start_time
    logger.info(f"Completed online augmentation pipeline in {total_time:.2f} seconds")
    
    return output_data

def find_input_files(config: Dict[str, Any]) -> List[str]:
    """
    Find input files based on configuration.
    
    Args:
        config: Configuration as a dictionary
        
    Returns:
        List of input file paths
    """
    # If file_path is specified, use it
    if config.get("input", {}).get("file_path"):
        file_path = config["input"]["file_path"]
        if os.path.exists(file_path):
            return [file_path]
        else:
            logger.warning(f"Specified input file not found: {file_path}")
            return []
    
    # Otherwise, use default_input_dir and file_pattern
    default_input_dir = config.get("input", {}).get("default_input_dir", "./data/output/")
    file_pattern = config.get("input", {}).get("file_pattern", "*_chunked.json")
    
    # Find files matching the pattern
    pattern = os.path.join(default_input_dir, file_pattern)
    files = glob.glob(pattern)
    
    if not files:
        logger.warning(f"No input files found matching pattern: {pattern}")
    
    return files

def main():
    """
    Main entry point for the online augmentation pipeline.
    """
    # Load environment variables from .env file
    # First try to load from the project root
    if os.path.exists(".env"):
        load_dotenv()
    # Then try to load from the pdf_extraction_cell directory
    elif os.path.exists("pdf_extraction_cell/.env"):
        load_dotenv("pdf_extraction_cell/.env")
    
    parser = argparse.ArgumentParser(description="Online Augmentation Pipeline")
    
    # Optional arguments
    parser.add_argument("--input", help="Path to the input file")
    parser.add_argument("--output", help="Path to save the output file (optional)")
    parser.add_argument("--config", default="online_augmentation/config/augmentation_config.json", 
                        help="Path to the configuration file")
    parser.add_argument("--batch", action="store_true", 
                        help="Process all files matching the pattern in the config")
    
    # Online search arguments
    parser.add_argument("--search", action="store_true",
                        help="Run online search after extraction")
    parser.add_argument("--api-key", help="OpenRouter API key for online search (overrides .env)")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Get API key from command line if provided, otherwise use from .env
    api_key = args.api_key if args.api_key else os.getenv("OPENROUTER_API_KEY")
    
    if args.batch:
        # Process all files matching the pattern
        input_files = find_input_files(config)
        logger.info(f"Found {len(input_files)} files to process")
        
        for input_file in input_files:
            process_file(
                input_file_path=input_file,
                output_file_path=None,
                config_path=args.config,
                run_online_search=args.search,
                api_key=api_key
            )
    else:
        # Process a single file
        input_file = args.input
        if not input_file:
            # If no input file is specified, use the first file matching the pattern
            input_files = find_input_files(config)
            if input_files:
                input_file = input_files[0]
            else:
                logger.error("No input file specified and no files found matching the pattern")
                sys.exit(1)
        
        process_file(
            input_file_path=input_file,
            output_file_path=args.output,
            config_path=args.config,
            run_online_search=args.search,
            api_key=api_key
        )

if __name__ == "__main__":
    main()
