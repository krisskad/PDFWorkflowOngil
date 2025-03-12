#!/usr/bin/env python3
"""
Text Post-Processing Script for PDF extraction results.

This script uses the Claude API to enhance the text portion of PDF extraction results
with advanced text analysis, entity extraction, summarization, and more.
"""

import os
import json
import logging
import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
env_path = script_dir / '.env'
if env_path.exists():
    load_dotenv(env_path)

# Get the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('text_post_processing.log', mode='w')
    ]
)

logger = logging.getLogger("pdf_extraction.text_post_process")

# Import the text post-processor
from modules.text_post_processor import TextPostProcessor

def load_config(config_path):
    """Load configuration from JSON file."""
    try:
        # If the path is not absolute, make it relative to the script directory
        if not os.path.isabs(config_path):
            config_path = os.path.join(SCRIPT_DIR, config_path)
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        raise

def process_file(input_file, output_file, config):
    """
    Process a single PDF extraction result file.
    
    Args:
        input_file: Path to the PDF extraction result JSON file
        output_file: Path to save the enhanced output
        config: Configuration dictionary
    """
    try:
        # If paths are not absolute, make them relative to the script directory
        if not os.path.isabs(input_file):
            if input_file.startswith('pdf_extraction_cell/'):
                relative_path = input_file[len('pdf_extraction_cell/'):]
                input_file = os.path.join(SCRIPT_DIR, relative_path)
            else:
                input_file = os.path.join(SCRIPT_DIR, input_file)
        
        if not os.path.isabs(output_file):
            if output_file.startswith('pdf_extraction_cell/'):
                relative_path = output_file[len('pdf_extraction_cell/'):]
                output_file = os.path.join(SCRIPT_DIR, relative_path)
            else:
                output_file = os.path.join(SCRIPT_DIR, output_file)
        
        # Load the input file
        with open(input_file, 'r') as f:
            parser_output = json.load(f)
        
        # Initialize the text post-processor
        post_processor = TextPostProcessor(config)
        
        # Process the text content
        logger.info(f"Processing text content from {input_file}")
        enhanced_output = post_processor.process_text_content(parser_output, output_file)
        
        # Print summary
        print("\nText Post-Processing Summary:")
        print(f"Input: {input_file}")
        print(f"Output: {output_file}")
        
        if "ai_enhancements" in enhanced_output:
            enhancements = enhanced_output["ai_enhancements"]
            print(f"Document type: {enhancements.get('document_type', {}).get('type', 'Unknown')}")
            
            if "page_level_analysis" in enhancements:
                page_count = len(enhancements["page_level_analysis"])
                print(f"Pages analyzed: {page_count}")
                
                # Count topics and entities at page level
                page_topics_count = sum(1 for page in enhancements["page_level_analysis"] if "topics" in page)
                page_entities_count = sum(1 for page in enhancements["page_level_analysis"] if "entities" in page)
                print(f"Pages with topic analysis: {page_topics_count}")
                print(f"Pages with entity extraction: {page_entities_count}")
            
            # Document-level analysis
            if "topics" in enhancements:
                topic_count = len(enhancements.get("topics", {}).get("main_topics", []))
                print(f"Document-level topics identified: {topic_count}")
            
            if "entities" in enhancements:
                entity_count = sum(len(entities) for entities in enhancements["entities"].values() 
                                  if isinstance(entities, list))
                print(f"Document-level entities extracted: {entity_count}")
            
            if "document_structure" in enhancements:
                section_count = len(enhancements["document_structure"])
                print(f"Sections identified: {section_count}")
            
            if "summary" in enhancements:
                print("Summary generated: Yes")
            
            if "key_information" in enhancements:
                print("Key information extracted: Yes")
            
            # List any custom analyses
            custom_analyses = [key for key in enhancements.keys() 
                              if key not in ["document_type", "topics", "entities", "document_structure", 
                                            "summary", "key_information", "page_level_analysis"]]
            if custom_analyses:
                print(f"Custom analyses: {', '.join(custom_analyses)}")
        
        return enhanced_output
    
    except Exception as e:
        logger.error(f"Error processing file {input_file}: {str(e)}")
        raise

def process_directory(input_dir, output_dir, config):
    """
    Process all JSON files in a directory.
    
    Args:
        input_dir: Directory containing PDF extraction result JSON files
        output_dir: Directory to save enhanced output files
        config: Configuration dictionary
    """
    # If paths are not absolute, make them relative to the script directory
    if not os.path.isabs(input_dir):
        if input_dir.startswith('pdf_extraction_cell/'):
            relative_path = input_dir[len('pdf_extraction_cell/'):]
            input_dir = os.path.join(SCRIPT_DIR, relative_path)
        else:
            input_dir = os.path.join(SCRIPT_DIR, input_dir)
    
    if not os.path.isabs(output_dir):
        if output_dir.startswith('pdf_extraction_cell/'):
            relative_path = output_dir[len('pdf_extraction_cell/'):]
            output_dir = os.path.join(SCRIPT_DIR, relative_path)
        else:
            output_dir = os.path.join(SCRIPT_DIR, output_dir)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all JSON files
    json_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.json')]
    
    if not json_files:
        logger.warning(f"No JSON files found in {input_dir}")
        return
    
    logger.info(f"Found {len(json_files)} JSON files to process")
    
    # Process each JSON file
    results = {}
    for json_file in json_files:
        input_path = os.path.join(input_dir, json_file)
        output_name = os.path.splitext(json_file)[0] + "_enhanced.json"
        output_path = os.path.join(output_dir, output_name)
        
        try:
            results[json_file] = process_file(input_path, output_path, config)
        except Exception as e:
            logger.error(f"Failed to process {json_file}: {str(e)}")
            continue
    
    return results

def main():
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Text Post-Processing for PDF extraction results")
    parser.add_argument("--input", "-i", help="Input JSON file or directory", required=True)
    parser.add_argument("--output", "-o", help="Output file or directory")
    parser.add_argument("--config", "-c", help="Configuration file", default="config/text_post_processing_config.json")
    parser.add_argument("--api-key", help="Claude API key (overrides config file)")
    parser.add_argument("--model", help="Claude model to use (overrides config file)")
    parser.add_argument("--company-name", help="Company name for targeted analysis")
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        # If the default config doesn't exist, create a minimal config
        if args.config == "config/text_post_processing_config.json":
            logger.warning("Default config file not found. Using minimal configuration.")
            config = {}
        else:
            raise
    
    # Override config with command-line arguments
    if args.api_key:
        config["claude_api_key"] = args.api_key
    if args.model:
        config["claude_model"] = args.model
    if args.company_name:
        config["company_name"] = args.company_name
    
    # Check for API key
    if not config.get("claude_api_key") and not os.environ.get("ANTHROPIC_API_KEY"):
        logger.error("Claude API key is required. Set it in config, as an argument, or in ANTHROPIC_API_KEY environment variable.")
        return 1
    
    # Process input
    input_path = args.input
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        # If no output specified, create default based on input
        if os.path.isfile(input_path) or input_path.endswith('.json'):
            base_name = os.path.splitext(input_path)[0]
            output_path = f"{base_name}_enhanced.json"
        else:
            # For directories, use a default enhanced directory
            input_dir = os.path.basename(input_path.rstrip('/\\'))
            output_path = os.path.join(os.path.dirname(input_path), f"{input_dir}_enhanced")
    
    # Check if the input path exists
    if os.path.exists(input_path):
        # Path exists as is, use it directly
        if os.path.isfile(input_path):
            process_file(input_path, output_path, config)
        elif os.path.isdir(input_path):
            process_directory(input_path, output_path, config)
        else:
            logger.error(f"Input path is neither a file nor a directory: {input_path}")
            return 1
    else:
        # Path doesn't exist as provided, try to resolve it
        
        # If it starts with pdf_extraction_cell/, remove that prefix
        if not os.path.isabs(input_path) and input_path.startswith('pdf_extraction_cell/'):
            input_path = input_path[len('pdf_extraction_cell/'):]
        
        # Try to resolve the path relative to the script directory
        script_relative_path = os.path.join(SCRIPT_DIR, input_path)
        
        if os.path.exists(script_relative_path):
            # Path exists relative to script directory
            if os.path.isfile(script_relative_path):
                if not args.output:
                    base_name = os.path.splitext(script_relative_path)[0]
                    output_path = f"{base_name}_enhanced.json"
                process_file(script_relative_path, output_path, config)
            elif os.path.isdir(script_relative_path):
                if not args.output:
                    input_dir = os.path.basename(script_relative_path.rstrip('/\\'))
                    output_path = os.path.join(os.path.dirname(script_relative_path), f"{input_dir}_enhanced")
                process_directory(script_relative_path, output_path, config)
            else:
                logger.error(f"Input path is neither a file nor a directory: {script_relative_path}")
                return 1
        else:
            # Path doesn't exist in any form we tried
            logger.error(f"Input path does not exist: {input_path}")
            return 1
    
    logger.info("Text post-processing complete")
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
