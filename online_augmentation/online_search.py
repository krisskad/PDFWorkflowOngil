"""
Online Search Module for the Online Augmentation Pipeline.
This script takes the output from main.py and performs online searches for specified entities
using the Perplexity LLM via OpenRouter API.
"""

import os
import sys
import logging
import argparse
import json
from typing import Dict, Any, List, Optional, Union
import time
from openai import OpenAI
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("online_augmentation.online_search")

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

def extract_requested_entities(
    augmented_data: Dict[str, Any], 
    entity_types: List[str],
    max_entities_per_type: int = 5
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extract the requested entity types from the augmented data with additional context.
    
    Args:
        augmented_data: The augmented data from main.py
        entity_types: List of entity types to extract
        max_entities_per_type: Maximum number of entities to extract per type
        
    Returns:
        Dictionary of entity types and their values with context
    """
    requested_entities = {}
    
    if "entities" not in augmented_data:
        logger.warning("No entities found in augmented data")
        return requested_entities
    
    for entity_type in entity_types:
        if entity_type in augmented_data["entities"]:
            entities = augmented_data["entities"][entity_type]
            
            # Extract entity values with context based on the structure
            entity_details = []
            
            if isinstance(entities, list):
                for entity in entities[:max_entities_per_type]:
                    # Create a detailed entity object
                    entity_detail = {}
                    
                    if isinstance(entity, dict):
                        # Extract the main entity name/text
                        if "text" in entity:
                            entity_detail["name"] = entity["text"]
                        elif "name" in entity:
                            entity_detail["name"] = entity["name"]
                        elif "section_name" in entity:
                            entity_detail["name"] = entity["section_name"]
                        elif "complete_name" in entity:
                            entity_detail["name"] = entity["complete_name"]
                        
                        # Extract additional context and information
                        for key, value in entity.items():
                            if key not in ["text", "name", "section_name", "complete_name"]:
                                entity_detail[key] = value
                        
                        # Extract location information if available
                        if "location" in entity:
                            entity_detail["location"] = entity["location"]
                        
                        # Extract context if available
                        if "context" in entity:
                            entity_detail["context"] = entity["context"]
                        elif "surrounding_text" in entity:
                            entity_detail["context"] = entity["surrounding_text"]
                        
                        # Extract confidence if available
                        if "confidence" in entity:
                            entity_detail["confidence"] = entity["confidence"]
                        
                        # Extract relationships if available
                        if "relationships" in entity:
                            entity_detail["relationships"] = entity["relationships"]
                        
                        # If we have a name, add this entity to our list
                        if "name" in entity_detail:
                            entity_details.append(entity_detail)
                    elif isinstance(entity, str):
                        entity_details.append({"name": entity})
            elif isinstance(entities, str):
                entity_details.append({"name": entities})
            
            if entity_details:
                requested_entities[entity_type] = entity_details
        else:
            logger.warning(f"Entity type '{entity_type}' not found in augmented data")
    
    return requested_entities

def format_entities_for_search(entities: Dict[str, List[Dict[str, Any]]]) -> str:
    """
    Format the entities with context for the search query.
    
    Args:
        entities: Dictionary of entity types and their detailed values
        
    Returns:
        Formatted search query with context
    """
    query_parts = []
    
    for entity_type, entity_details in entities.items():
        if entity_details:
            # Start with the entity type header
            entity_section = [f"## {entity_type}:"]
            
            # Add each entity with its context
            for entity in entity_details:
                # Start with the entity name
                entity_part = [f"- {entity['name']}"]
                
                # Add location if available
                if "location" in entity:
                    if isinstance(entity["location"], dict):
                        location_str = ", ".join(f"{k}: {v}" for k, v in entity["location"].items())
                        entity_part.append(f"  Location: {location_str}")
                    else:
                        entity_part.append(f"  Location: {entity['location']}")
                
                # Add context if available
                if "context" in entity:
                    entity_part.append(f"  Context: {entity['context']}")
                
                # Add relationships if available
                if "relationships" in entity and entity["relationships"]:
                    if isinstance(entity["relationships"], list):
                        rel_str = ", ".join(str(r) for r in entity["relationships"])
                        entity_part.append(f"  Related to: {rel_str}")
                    else:
                        entity_part.append(f"  Related to: {entity['relationships']}")
                
                # Add any other relevant details
                for key, value in entity.items():
                    if key not in ["name", "location", "context", "relationships", "confidence"] and value:
                        entity_part.append(f"  {key.replace('_', ' ').title()}: {value}")
                
                # Join this entity's details and add to the section
                entity_section.append("\n".join(entity_part))
            
            # Join all entities in this type and add to query parts
            query_parts.append("\n".join(entity_section))
    
    # Join all entity sections with double newlines
    return "\n\n".join(query_parts)

def perform_perplexity_search(
    search_query: str,
    api_key: str,
    llm_model: str = "perplexity/sonar-reasoning-pro",
    site_url: str = "https://example.com",
    site_name: str = "PDF Workflow"
) -> str:
    """
    Perform a search using the Perplexity LLM via OpenRouter API with enhanced context.
    
    Args:
        search_query: The search query with context
        api_key: The OpenRouter API key
        llm_model: The LLM model to use
        site_url: The site URL for rankings on openrouter.ai
        site_name: The site name for rankings on openrouter.ai
        
    Returns:
        The detailed search results
    """
    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        
        # Construct the prompt for the search with enhanced detail requirements
        prompt = f"""Please search for comprehensive information about the following entities and provide detailed, up-to-date information. I've included context and additional details where available:

{search_query}

For each entity, please provide:

1. A detailed description including historical background
2. Geographical context and location details where relevant
3. Current status and recent developments
4. Key statistics or metrics associated with the entity
5. Relationships and connections to other entities (both in this list and significant external relationships)
6. Relevant regulatory or legal information if applicable
7. Any controversies, challenges, or notable achievements
8. For locations or geographical entities, include information about:
   - Demographics
   - Economic significance
   - Infrastructure details
   - Environmental factors
9. For organizations, include:
   - Leadership information
   - Market position
   - Notable products or services
10. For people, include:
    - Professional background
    - Notable contributions or achievements
    - Current roles and affiliations

Please organize the information clearly and provide source references where possible. If certain information isn't available or relevant for a particular entity, focus on the aspects that are most important.
"""
        
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": site_url,
                "X-Title": site_name,
            },
            extra_body={},
            model=llm_model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        return completion.choices[0].message.content
    except Exception as e:
        logger.error(f"Error performing Perplexity search: {e}")
        return f"Error performing search: {str(e)}"

def process_file(
    input_file_path: str,
    entity_types: List[str],
    api_key: str,
    output_file_path: str = None,
    max_entities_per_type: int = 5,
    llm_model: str = "perplexity/sonar-reasoning-pro",
    site_url: str = "https://example.com",
    site_name: str = "PDF Workflow"
) -> Dict[str, Any]:
    """
    Process an augmented file through the online search pipeline.
    
    Args:
        input_file_path: Path to the input file (output from main.py)
        entity_types: List of entity types to search for
        api_key: The OpenRouter API key
        output_file_path: Path to save the output file (optional)
        max_entities_per_type: Maximum number of entities to extract per type
        llm_model: The LLM model to use
        site_url: The site URL for rankings on openrouter.ai
        site_name: The site name for rankings on openrouter.ai
        
    Returns:
        Processed data as a dictionary
    """
    start_time = time.time()
    logger.info(f"Starting online search for file: {input_file_path}")
    
    # If output_file_path is not provided, construct it from the input path
    if not output_file_path:
        input_file_name = os.path.basename(input_file_path)
        output_dir = os.path.dirname(input_file_path)
        output_file_name = input_file_name.replace("_augmented.json", "_searched.json")
        output_file_path = os.path.join(output_dir, output_file_name)
    
    logger.info(f"Input file path: {input_file_path}")
    logger.info(f"Output file path: {output_file_path}")
    
    # Read input file
    augmented_data = read_json(input_file_path)
    logger.info(f"Loaded augmented data")
    
    # Extract requested entities
    requested_entities = extract_requested_entities(
        augmented_data, 
        entity_types,
        max_entities_per_type
    )
    logger.info(f"Extracted requested entities: {requested_entities.keys()}")
    
    # Format entities for search
    search_query = format_entities_for_search(requested_entities)
    logger.info(f"Formatted search query:\n{search_query}")
    
    # Perform search
    if search_query:
        search_results = perform_perplexity_search(
            search_query, 
            api_key,
            llm_model,
            site_url,
            site_name
        )
        logger.info("Completed search")
        
        # Add search results to output data
        output_data = augmented_data.copy()
        output_data["online_search"] = {
            "query": search_query,
            "results": search_results,
            "searched_entities": requested_entities,
            "timestamp": time.time()
        }
        
        # Write output to file
        write_json(output_data, output_file_path)
        logger.info(f"Wrote output to {output_file_path}")
    else:
        logger.warning("No entities found for search query")
        output_data = augmented_data.copy()
        output_data["online_search"] = {
            "error": "No entities found for search query",
            "timestamp": time.time()
        }
        
        # Write output to file
        write_json(output_data, output_file_path)
        logger.info(f"Wrote output to {output_file_path}")
    
    # Log processing time
    processing_time = time.time() - start_time
    logger.info(f"Completed online search in {processing_time:.2f} seconds")
    
    return output_data

def main():
    """
    Main entry point for the online search pipeline.
    """
    # Load environment variables from .env file
    # First try to load from the project root
    if os.path.exists(".env"):
        load_dotenv()
    # Then try to load from the pdf_extraction_cell directory
    elif os.path.exists("pdf_extraction_cell/.env"):
        load_dotenv("pdf_extraction_cell/.env")
    
    # Get API key from environment variable
    env_api_key = os.getenv("OPENROUTER_API_KEY")
    
    parser = argparse.ArgumentParser(description="Online Search Pipeline")
    
    # Required arguments (api-key is now optional if provided in .env)
    parser.add_argument("--input", required=True, help="Path to the input file (output from main.py)")
    parser.add_argument("--api-key", help="OpenRouter API key (overrides .env)")
    
    # Optional arguments
    parser.add_argument("--config", default="./online_augmentation/config/augmentation_config.json",
                        help="Path to the configuration file")
    parser.add_argument("--output", help="Path to save the output file (optional)")
    parser.add_argument("--entity-types", nargs="+",
                        help="Entity types to search for (space-separated, overrides config)")
    parser.add_argument("--max-entities", type=int,
                        help="Maximum number of entities to extract per type (overrides config)")
    parser.add_argument("--llm-model",
                        help="LLM model to use (overrides config)")
    parser.add_argument("--site-url",
                        help="Site URL for rankings on openrouter.ai (overrides config)")
    parser.add_argument("--site-name",
                        help="Site name for rankings on openrouter.ai (overrides config)")
    
    args = parser.parse_args()
    
    # Use API key from command line if provided, otherwise use from .env
    api_key = args.api_key if args.api_key else env_api_key
    
    if not api_key:
        logger.error("No API key provided. Please provide an API key via --api-key or set OPENROUTER_API_KEY in .env")
        sys.exit(1)
    
    # Load configuration
    config = load_config(args.config)
    online_search_config = config.get("online_search", {})
    
    # Use command-line arguments if provided, otherwise use configuration
    entity_types = args.entity_types if args.entity_types is not None else online_search_config.get("entity_types", ["Organizations", "People", "Locations"])
    max_entities = args.max_entities if args.max_entities is not None else online_search_config.get("max_entities_per_type", 5)
    llm_model = args.llm_model if args.llm_model is not None else online_search_config.get("llm_model", "perplexity/sonar-reasoning-pro")
    site_url = args.site_url if args.site_url is not None else online_search_config.get("site_url", "https://example.com")
    site_name = args.site_name if args.site_name is not None else online_search_config.get("site_name", "PDF Workflow")
    
    # Check if online search is enabled
    if not online_search_config.get("enabled", True):
        logger.warning("Online search is disabled in the configuration")
        return
    
    logger.info(f"Using configuration: entity_types={entity_types}, max_entities={max_entities}, llm_model={llm_model}")
    
    process_file(
        input_file_path=args.input,
        entity_types=entity_types,
        api_key=api_key,
        output_file_path=args.output,
        max_entities_per_type=max_entities,
        llm_model=llm_model,
        site_url=site_url,
        site_name=site_name
    )

if __name__ == "__main__":
    main()
