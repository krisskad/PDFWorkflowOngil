#!/usr/bin/env python3
"""
Table Post-Processing Script for PDF extraction results.

This script uses the TablePostProcessor to enhance table extraction results
with contextual information from surrounding text.
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
        logging.FileHandler('table_post_processing.log', mode='w')
    ]
)

logger = logging.getLogger("pdf_extraction.table_post_process")

# Import the table post-processor
from modules.table_post_processor import TablePostProcessor

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

def process_file(input_file, output_file, config, text_file=None, company_name=None):
    """
    Process a single table extraction result file.
    
    Args:
        input_file: Path to the table extraction result JSON file
        output_file: Path to save the enhanced output
        config: Configuration dictionary
        text_file: Optional path to text extraction result for context
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
        
        if text_file and not os.path.isabs(text_file):
            if text_file.startswith('pdf_extraction_cell/'):
                relative_path = text_file[len('pdf_extraction_cell/'):]
                text_file = os.path.join(SCRIPT_DIR, relative_path)
            else:
                text_file = os.path.join(SCRIPT_DIR, text_file)
        
        # Load the input table file
        with open(input_file, 'r') as f:
            tables = json.load(f)
        
        # Load text file for context if provided
        enhanced_text = None
        if text_file and os.path.exists(text_file):
            try:
                with open(text_file, 'r') as f:
                    enhanced_text = json.load(f)
                logger.info(f"Loaded text context from {text_file}")
            except Exception as e:
                logger.warning(f"Error loading text context file: {str(e)}")
        
        # Initialize the table post-processor
        post_processor = TablePostProcessor(config, company_name)
        
        # Process the tables
        logger.info(f"Processing tables from {input_file}")
        enhanced_tables = post_processor.process_tables(tables, enhanced_text, output_file)
        
        # Print summary
        print("\nTable Post-Processing Summary:")
        print(f"Input: {input_file}")
        print(f"Output: {output_file}")
        print(f"Tables processed: {len(tables)}")
        print(f"Tables after processing: {len(enhanced_tables)}")
        
        # Print details about enhancements
        enhancements = []
        if any(table.get("metadata", {}).get("headers_cleaned", False) for table in enhanced_tables):
            enhancements.append("Headers cleaned")
        if any(table.get("metadata", {}).get("alignment_fixed", False) for table in enhanced_tables):
            enhancements.append("Alignment fixed")
        if any(table.get("metadata", {}).get("has_merged_cells", False) for table in enhanced_tables):
            enhancements.append("Merged cells detected")
        if any(table.get("metadata", {}).get("title", False) for table in enhanced_tables):
            enhancements.append("Titles extracted")
        if any(table.get("metadata", {}).get("footnotes", False) for table in enhanced_tables):
            enhancements.append("Footnotes extracted")
        if any(table.get("metadata", {}).get("claude_enhanced", False) for table in enhanced_tables):
            enhancements.append("Claude AI enhancements applied")
        if any(table.get("metadata", {}).get("data_types_standardized", False) for table in enhanced_tables):
            enhancements.append("Data types standardized")
        
        if enhancements:
            print(f"Enhancements applied: {', '.join(enhancements)}")
        else:
            print("No significant enhancements were needed")
        
        return enhanced_tables
    
    except Exception as e:
        logger.error(f"Error processing file {input_file}: {str(e)}")
        raise

def process_directory(input_dir, output_dir, config, text_dir=None, company_name=None):
    """
    Process all JSON table files in a directory.
    
    Args:
        input_dir: Directory containing table extraction result JSON files
        output_dir: Directory to save enhanced output files
        config: Configuration dictionary
        text_dir: Optional directory containing text extraction results for context
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
    
    if text_dir and not os.path.isabs(text_dir):
        if text_dir.startswith('pdf_extraction_cell/'):
            relative_path = text_dir[len('pdf_extraction_cell/'):]
            text_dir = os.path.join(SCRIPT_DIR, relative_path)
        else:
            text_dir = os.path.join(SCRIPT_DIR, text_dir)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all JSON files that look like table files
    json_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.json') and 'table' in f.lower()]
    
    if not json_files:
        logger.warning(f"No table JSON files found in {input_dir}")
        return
    
    logger.info(f"Found {len(json_files)} table JSON files to process")
    
    # Process each JSON file
    results = {}
    for json_file in json_files:
        input_path = os.path.join(input_dir, json_file)
        output_name = os.path.splitext(json_file)[0] + "_enhanced.json"
        output_path = os.path.join(output_dir, output_name)
        
        # Look for matching text file for context
        text_file = None
        if text_dir:
            base_name = json_file.replace('_tables', '').replace('_table', '')
            potential_text_files = [
                os.path.join(text_dir, base_name),
                os.path.join(text_dir, base_name.replace('.json', '_enhanced.json'))
            ]
            for potential_file in potential_text_files:
                if os.path.exists(potential_file):
                    text_file = potential_file
                    break
        
        try:
            results[json_file] = process_file(input_path, output_path, config, text_file, company_name)
        except Exception as e:
            logger.error(f"Failed to process {json_file}: {str(e)}")
            continue
    
    return results

def main():
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Table Post-Processing for PDF extraction results")
    parser.add_argument("--input", "-i", help="Input JSON file or directory with table data", required=True)
    parser.add_argument("--output", "-o", help="Output file or directory")
    parser.add_argument("--config", "-c", help="Configuration file", default="config/table_post_processing_config.json")
    parser.add_argument("--text", "-t", help="Text extraction file or directory for context")
    parser.add_argument("--api-key", help="Claude API key (overrides config file)")
    parser.add_argument("--model", help="Claude model to use (overrides config file)")
    parser.add_argument("--company", help="Company name for context-aware processing")
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        # If the default config doesn't exist, create a minimal config
        if args.config == "config/table_post_processing_config.json":
            logger.warning("Default config file not found. Using minimal configuration.")
            config = {
                "fix_headers": True,
                "fix_alignment": True,
                "detect_merged_cells": True,
                "extract_table_titles": True,
                "extract_footnotes": True,
                "dedup_tables": True,
                "clean_empty_rows_cols": True,
                "context_window_size": 1,
                "use_claude_api": False
            }
        else:
            raise
    
    # Override config with command-line arguments
    if args.api_key:
        config["claude_api_key"] = args.api_key
        config["use_claude_api"] = True
    if args.model:
        config["claude_model"] = args.model
    
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
            process_file(input_path, output_path, config, args.text, args.company)
        elif os.path.isdir(input_path):
            process_directory(input_path, output_path, config, args.text, args.company)
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
                process_file(script_relative_path, output_path, config, args.text, args.company)
            elif os.path.isdir(script_relative_path):
                if not args.output:
                    input_dir = os.path.basename(script_relative_path.rstrip('/\\'))
                    output_path = os.path.join(os.path.dirname(script_relative_path), f"{input_dir}_enhanced")
                process_directory(script_relative_path, output_path, config, args.text, args.company)
            else:
                logger.error(f"Input path is neither a file nor a directory: {script_relative_path}")
                return 1
        else:
            # Path doesn't exist in any form we tried
            logger.error(f"Input path does not exist: {input_path}")
            return 1
    
    logger.info("Table post-processing complete")
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
