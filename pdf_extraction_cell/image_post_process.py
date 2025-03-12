#!/usr/bin/env python3
"""
Image Post-Processing Script for PDF extraction results.

This script uses Gemini Flash Lite 2 via OpenRouter to identify if extracted images are charts
and extract tabular data from them if they are.
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
        logging.FileHandler('image_post_processing.log', mode='w')
    ]
)

logger = logging.getLogger("pdf_extraction.image_post_process")

# Import the image post-processor
from modules.image_post_processor import ImagePostProcessor

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

def process_file(input_file, output_file, config, company_name=None):
    """
    Process a single image metadata JSON file.
    
    Args:
        input_file: Path to the image metadata JSON file
        output_file: Path to save the enhanced output
        config: Configuration dictionary
        company_name: Optional company name for context-aware processing
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
            images_data = json.load(f)
        
        # Initialize the image post-processor
        post_processor = ImagePostProcessor(config, company_name)
        
        # Process the images
        logger.info(f"Processing images from {input_file}")
        processed_data = post_processor.process_images(images_data, output_file)
        
        # Print summary
        print("\nImage Post-Processing Summary:")
        print(f"Input: {input_file}")
        print(f"Output: {output_file}")
        
        summary = processed_data.get("summary", {})
        total_images = summary.get("total_images", 0)
        processed_images = summary.get("processed_images", 0)
        charts_detected = summary.get("charts_detected", 0)
        tables_extracted = summary.get("tables_extracted", 0)
        
        print(f"Total images: {total_images}")
        print(f"Images processed: {processed_images}")
        print(f"Charts detected: {charts_detected}")
        print(f"Tables extracted: {tables_extracted}")
        
        if charts_detected > 0:
            print("\nDetected Charts:")
            for i, img in enumerate(processed_data.get("processed_images", [])):
                if img.get("is_chart", False):
                    chart_type = img.get("chart_type", "Unknown")
                    confidence = img.get("confidence", 0)
                    image_path = img.get("path", "Unknown path")
                    print(f"  {i+1}. {os.path.basename(image_path)}: {chart_type} (confidence: {confidence:.2f})")
                    
                    if "extracted_table" in img:
                        table = img["extracted_table"]
                        headers = table.get("headers", [])
                        data_rows = len(table.get("data", []))
                        print(f"     Table extracted: {data_rows} rows, {len(headers)} columns")
        
        return processed_data
    
    except Exception as e:
        logger.error(f"Error processing file {input_file}: {str(e)}")
        raise

def process_directory(input_dir, output_dir, config, company_name=None):
    """
    Process all image JSON files in a directory.
    
    Args:
        input_dir: Directory containing image metadata JSON files
        output_dir: Directory to save enhanced output files
        config: Configuration dictionary
        company_name: Optional company name for context-aware processing
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
    
    # Get all JSON files that look like image metadata files
    json_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.json') and 'image' in f.lower()]
    
    if not json_files:
        logger.warning(f"No image JSON files found in {input_dir}")
        return
    
    logger.info(f"Found {len(json_files)} image JSON files to process")
    
    # Process each JSON file
    results = {}
    for json_file in json_files:
        input_path = os.path.join(input_dir, json_file)
        output_name = os.path.splitext(json_file)[0] + "_charts.json"
        output_path = os.path.join(output_dir, output_name)
        
        try:
            results[json_file] = process_file(input_path, output_path, config, company_name)
        except Exception as e:
            logger.error(f"Failed to process {json_file}: {str(e)}")
            continue
    
    return results

def main():
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Image Post-Processing for PDF extraction results")
    parser.add_argument("--input", "-i", help="Input JSON file or directory with image metadata", required=True)
    parser.add_argument("--output", "-o", help="Output file or directory")
    parser.add_argument("--config", "-c", help="Configuration file", default="config/image_post_processing_config.json")
    parser.add_argument("--api-key", help="OpenRouter API key (overrides config file)")
    parser.add_argument("--model", help="Gemini model to use (overrides config file)")
    parser.add_argument("--company", help="Company name for context-aware processing")
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        # If the default config doesn't exist, create a minimal config
        if args.config == "config/image_post_processing_config.json":
            logger.warning("Default config file not found. Using minimal configuration.")
            config = {
                "chart_detection": {
                    "enabled": True,
                    "confidence_threshold": 0.7
                },
                "table_extraction": {
                    "enabled": True,
                    "include_headers": True,
                    "standardize_data_types": True
                },
                "use_openrouter_api": True,
                "gemini_model": "google/gemini-2.0-flash-lite-001"
            }
        else:
            raise
    
    # Override config with command-line arguments
    if args.api_key:
        config["openrouter_api_key"] = args.api_key
        config["use_openrouter_api"] = True
    if args.model:
        config["gemini_model"] = args.model
    
    # Check for API key
    if not config.get("openrouter_api_key") and not os.environ.get("OPENROUTER_API_KEY"):
        logger.error("OpenRouter API key is required. Set it in config, as an argument, or in OPENROUTER_API_KEY environment variable.")
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
            output_path = f"{base_name}_charts.json"
        else:
            # For directories, use a default charts directory
            input_dir = os.path.basename(input_path.rstrip('/\\'))
            output_path = os.path.join(os.path.dirname(input_path), f"{input_dir}_charts")
    
    # Check if the input path exists
    if os.path.exists(input_path):
        # Path exists as is, use it directly
        if os.path.isfile(input_path):
            process_file(input_path, output_path, config, args.company)
        elif os.path.isdir(input_path):
            process_directory(input_path, output_path, config, args.company)
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
                    output_path = f"{base_name}_charts.json"
                process_file(script_relative_path, output_path, config, args.company)
            elif os.path.isdir(script_relative_path):
                if not args.output:
                    input_dir = os.path.basename(script_relative_path.rstrip('/\\'))
                    output_path = os.path.join(os.path.dirname(script_relative_path), f"{input_dir}_charts")
                process_directory(script_relative_path, output_path, config, args.company)
            else:
                logger.error(f"Input path is neither a file nor a directory: {script_relative_path}")
                return 1
        else:
            # Path doesn't exist in any form we tried
            logger.error(f"Input path does not exist: {input_path}")
            return 1
    
    logger.info("Image post-processing complete")
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
