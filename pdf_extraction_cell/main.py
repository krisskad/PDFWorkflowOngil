#!/usr/bin/env python3
"""
Main script for PDF Extraction Cell.

This script demonstrates how to use the PDF Extraction Cell framework
to extract and organize content from PDF documents.
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
        logging.FileHandler('extraction.log', mode='w')
    ]
)

logger = logging.getLogger("pdf_extraction")

# Import the extraction classes
from modules.pdf_parser import PDFParser
from modules.table_extractor import TableExtractor
from modules.image_extractor import ImageExtractor
from modules.text_post_processor import TextPostProcessor
from modules.table_post_processor import TablePostProcessor
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

def process_pdf(pdf_path, output_dir, config, use_direct_camelot=False, 
               apply_post_processing=False, post_process_config=None, 
               apply_table_post_processing=False, table_post_process_config=None,
               apply_image_post_processing=False, image_post_process_config=None,
               api_key=None):
    """
    Process a single PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save output files
        config: Configuration dictionary
        use_direct_camelot: If True, use Camelot directly for table detection without relying on pre-identified locations
    """
    try:
        # If paths are not absolute, make them relative to the script directory
        if not os.path.isabs(pdf_path):
            # Check if the path already starts with the script directory name
            if pdf_path.startswith('pdf_extraction_cell/'):
                # Remove the script directory name from the path
                relative_path = pdf_path[len('pdf_extraction_cell/'):]
                pdf_path = os.path.join(SCRIPT_DIR, relative_path)
            else:
                pdf_path = os.path.join(SCRIPT_DIR, pdf_path)
        
        if not os.path.isabs(output_dir):
            if output_dir.startswith('pdf_extraction_cell/'):
                relative_path = output_dir[len('pdf_extraction_cell/'):]
                output_dir = os.path.join(SCRIPT_DIR, relative_path)
            else:
                output_dir = os.path.join(SCRIPT_DIR, output_dir)
        # Initialize extractors with configuration
        parser = PDFParser(config)
        table_extractor = TableExtractor(config)
        image_extractor = ImageExtractor(config)
        
        # Extract all content
        logger.info(f"Processing PDF: {pdf_path}")
        result = parser.extract_all(pdf_path)
        
        # Extract tables using the parser output or direct Camelot
        if config.get("extraction_settings", {}).get("extract_tables", True):
            logger.info(f"Extracting tables from {pdf_path}")
            tables = table_extractor.extract_tables_from_parser_output(pdf_path, result, use_direct_camelot)
            result["tables"] = tables
            extraction_method = "direct Camelot" if use_direct_camelot else "parser-guided extraction"
            logger.info(f"Extracted {len(tables)} tables from {pdf_path} using {extraction_method}")
        
        # Extract images using the parser output
        if config.get("extraction_settings", {}).get("extract_images", True):
            logger.info(f"Extracting images from {pdf_path}")
            images_output_dir = os.path.join(output_dir, "extracted_images")
            images = image_extractor.extract_images_from_parser_output(pdf_path, result, output_dir=images_output_dir)
            result["images"] = images
            logger.info(f"Extracted {len(images)} images from {pdf_path}")
        
        # Create output filename
        pdf_name = os.path.basename(pdf_path)
        output_name = os.path.splitext(pdf_name)[0] + ".json"
        output_path = os.path.join(output_dir, output_name)
        
        # Apply text post-processing if requested
        if apply_post_processing:
            try:
                # Load post-processing config if provided
                if post_process_config:
                    pp_config = load_config(post_process_config)
                else:
                    pp_config = {}
                
                # Override with API key if provided
                if api_key:
                    pp_config["claude_api_key"] = api_key
                
                # Initialize text post-processor
                logger.info("Applying text post-processing with Claude API")
                text_processor = TextPostProcessor(pp_config)
                
                # Process the text content
                enhanced_output_path = os.path.splitext(output_path)[0] + "_enhanced.json"
                enhanced_result = text_processor.process_text_content(result, enhanced_output_path)
                
                # Update the result with AI enhancements
                result["ai_enhancements"] = enhanced_result.get("ai_enhancements", {})
                
                logger.info(f"Text post-processing complete, enhanced output saved to {enhanced_output_path}")
                
                # Print AI enhancement summary
                enhancements = result.get("ai_enhancements", {})
                print("\nText AI Enhancement Summary:")
                print(f"Document type: {enhancements.get('document_type', {}).get('type', 'Unknown')}")
                
                if "entities" in enhancements:
                    entity_count = sum(len(entities) for entities in enhancements["entities"].values() 
                                      if isinstance(entities, list))
                    print(f"Entities extracted: {entity_count}")
                
                if "document_structure" in enhancements:
                    section_count = len(enhancements["document_structure"])
                    print(f"Sections identified: {section_count}")
                
                if "summary" in enhancements:
                    print("Summary generated: Yes")
                
                if "key_information" in enhancements:
                    print("Key information extracted: Yes")
                
            except Exception as e:
                logger.error(f"Error during text post-processing: {str(e)}")
                print(f"\nWarning: Text post-processing failed: {str(e)}")
                print("Continuing with basic extraction results only.")
        
        # Apply table post-processing if requested
        if apply_table_post_processing and "tables" in result and result["tables"]:
            try:
                # Load table post-processing config if provided
                if table_post_process_config:
                    table_pp_config = load_config(table_post_process_config)
                else:
                    table_pp_config = {}
                
                # Override with API key if provided
                if api_key:
                    table_pp_config["claude_api_key"] = api_key
                
                # Initialize table post-processor
                logger.info("Applying table post-processing with Claude API")
                table_processor = TablePostProcessor(table_pp_config)
                
                # Process the tables
                tables_output_path = os.path.join(output_dir, os.path.splitext(pdf_name)[0] + "_tables_enhanced.json")
                enhanced_tables = table_processor.process_tables(
                    result["tables"], 
                    enhanced_result if apply_post_processing else None,
                    tables_output_path
                )
                
                # Update the result with enhanced tables
                result["tables"] = enhanced_tables
                
                logger.info(f"Table post-processing complete, enhanced tables saved to {tables_output_path}")
                
                # Print table enhancement summary
                table_enhancements_count = sum(1 for table in enhanced_tables if "ai_enhancements" in table)
                if table_enhancements_count > 0:
                    print("\nTable AI Enhancement Summary:")
                    print(f"Tables enhanced: {table_enhancements_count}")
                    
                    # Count specific enhancements
                    tables_with_summary = sum(1 for table in enhanced_tables if table.get("ai_enhancements", {}).get("summary"))
                    tables_with_keywords = sum(1 for table in enhanced_tables if table.get("ai_enhancements", {}).get("keywords"))
                    tables_with_insights = sum(1 for table in enhanced_tables if table.get("ai_enhancements", {}).get("insights"))
                    tables_with_type = sum(1 for table in enhanced_tables if table.get("ai_enhancements", {}).get("table_type"))
                    
                    if tables_with_summary > 0:
                        print(f"Tables with summaries: {tables_with_summary}")
                    if tables_with_keywords > 0:
                        print(f"Tables with keywords: {tables_with_keywords}")
                    if tables_with_insights > 0:
                        print(f"Tables with insights: {tables_with_insights}")
                    if tables_with_type > 0:
                        print(f"Tables with type detection: {tables_with_type}")
                
            except Exception as e:
                logger.error(f"Error during table post-processing: {str(e)}")
                print(f"\nWarning: Table post-processing failed: {str(e)}")
                print("Continuing without table enhancements.")
        
        # Apply image post-processing if requested
        if apply_image_post_processing and "images" in result and result["images"]:
            try:
                # Load image post-processing config if provided
                if image_post_process_config:
                    image_pp_config = load_config(image_post_process_config)
                else:
                    image_pp_config = {}
                
                # Override with API key if provided
                if api_key:
                    image_pp_config["claude_api_key"] = api_key
                
                # Initialize image post-processor
                logger.info("Applying image post-processing with Claude API")
                image_processor = ImagePostProcessor(image_pp_config)
                
                # Process the images
                images_output_path = os.path.join(output_dir, "extracted_images", os.path.splitext(pdf_name)[0] + "_charts.json")
                processed_images_result = image_processor.process_images(result["images"], images_output_path)
                
                # Update the result with processed images
                result["images"] = processed_images_result.get("processed_images", result["images"])
                
                logger.info(f"Image post-processing complete, processed images saved to {images_output_path}")
                
                # Print image enhancement summary
                summary = processed_images_result.get("summary", {})
                charts_detected = summary.get("charts_detected", 0)
                tables_extracted = summary.get("tables_extracted", 0)
                
                if charts_detected > 0:
                    print("\nImage AI Enhancement Summary:")
                    print(f"Charts detected: {charts_detected}")
                    print(f"Tables extracted from charts: {tables_extracted}")
                    
                    # Count specific enhancements
                    charts_with_summary = sum(1 for img in result["images"] if img.get("ai_enhancements", {}).get("summary"))
                    charts_with_keywords = sum(1 for img in result["images"] if img.get("ai_enhancements", {}).get("keywords"))
                    charts_with_insights = sum(1 for img in result["images"] if img.get("ai_enhancements", {}).get("insights"))
                    charts_with_trends = sum(1 for img in result["images"] if img.get("ai_enhancements", {}).get("trends"))
                    
                    if charts_with_summary > 0:
                        print(f"Charts with summaries: {charts_with_summary}")
                    if charts_with_keywords > 0:
                        print(f"Charts with keywords: {charts_with_keywords}")
                    if charts_with_insights > 0:
                        print(f"Charts with insights: {charts_with_insights}")
                    if charts_with_trends > 0:
                        print(f"Charts with trend analysis: {charts_with_trends}")
                
            except Exception as e:
                logger.error(f"Error during image post-processing: {str(e)}")
                print(f"\nWarning: Image post-processing failed: {str(e)}")
                print("Continuing without image enhancements.")
        
        # Save results to JSON file (with or without enhancements)
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Final results saved to {output_path}")
        
        # Print summary
        print("\nExtraction Summary:")
        print(f"PDF: {pdf_path}")
        print(f"Pages: {result['metadata'].get('page_count', 0)}")
        print(f"Title: {result['metadata'].get('title', 'Unknown')}")
        print(f"Author: {result['metadata'].get('author', 'Unknown')}")
        print(f"Tables detected: {len(result['visual_elements'].get('tables', []))}")
        print(f"Tables extracted: {len(result.get('tables', []))}")
        print(f"Images detected: {len(result['visual_elements'].get('images', []))}")
        print(f"Images extracted: {len(result.get('images', []))}")
        print(f"Output: {output_path}")
        
        return result
    
    except Exception as e:
        logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
        raise

def process_directory(input_dir, output_dir, config, use_direct_camelot=False, 
                     apply_post_processing=False, post_process_config=None,
                     apply_table_post_processing=False, table_post_process_config=None,
                     apply_image_post_processing=False, image_post_process_config=None,
                     api_key=None):
    """
    Process all PDF files in a directory.
    
    Args:
        input_dir: Directory containing PDF files
        output_dir: Directory to save output files
        config: Configuration dictionary
        use_direct_camelot: If True, use Camelot directly for table detection without relying on pre-identified locations
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
    
    # Get all PDF files
    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        logger.warning(f"No PDF files found in {input_dir}")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    # Process each PDF
    results = {}
    for pdf_file in pdf_files:
        pdf_path = os.path.join(input_dir, pdf_file)
        try:
            results[pdf_file] = process_pdf(pdf_path, output_dir, config, use_direct_camelot, 
                                           apply_post_processing, post_process_config,
                                           apply_table_post_processing, table_post_process_config,
                                           apply_image_post_processing, image_post_process_config,
                                           api_key)
        except Exception as e:
            logger.error(f"Failed to process {pdf_file}: {str(e)}")
            continue
    
    return results

def main():
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="PDF Extraction Cell")
    parser.add_argument("--input", "-i", help="Input PDF file or directory", required=True)
    parser.add_argument("--output", "-o", help="Output directory", default="data/output")
    parser.add_argument("--config", "-c", help="Configuration file", default="config/extraction_config.json")
    parser.add_argument("--direct-camelot", "-d", action="store_true", 
                        help="Use Camelot directly for table detection without relying on pre-identified locations")
    parser.add_argument("--post-process", "-p", action="store_true",
                        help="Apply text post-processing using Claude API")
    parser.add_argument("--post-process-config", "-pc", 
                        help="Configuration file for text post-processing", 
                        default="config/text_post_processing_config.json")
    parser.add_argument("--table-post-process", "-tp", action="store_true",
                        help="Apply table post-processing using Claude API")
    parser.add_argument("--table-post-process-config", "-tpc", 
                        help="Configuration file for table post-processing", 
                        default="config/table_post_processing_config.json")
    parser.add_argument("--image-post-process", "-ip", action="store_true",
                        help="Apply image post-processing using Claude API")
    parser.add_argument("--image-post-process-config", "-ipc", 
                        help="Configuration file for image post-processing", 
                        default="config/image_post_processing_config.json")
    parser.add_argument("--api-key", 
                        help="Claude API key for post-processing (overrides config)")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Process input
    input_path = args.input
    output_dir = args.output
    
    # Handle the input path
    original_input_path = input_path
    
    # Get direct Camelot flag
    use_direct_camelot = args.direct_camelot
    
    # Get post-processing parameters
    apply_post_processing = args.post_process
    post_process_config = args.post_process_config if args.post_process else None
    
    # Get table post-processing parameters
    apply_table_post_processing = args.table_post_process
    table_post_process_config = args.table_post_process_config if args.table_post_process else None
    
    # Get image post-processing parameters
    apply_image_post_processing = args.image_post_process
    image_post_process_config = args.image_post_process_config if args.image_post_process else None
    
    api_key = args.api_key
    
    # Check if the path exists as provided
    if os.path.exists(input_path):
        # Path exists as is, use it directly
        if os.path.isfile(input_path):
            process_pdf(input_path, output_dir, config, use_direct_camelot, 
                       apply_post_processing, post_process_config,
                       apply_table_post_processing, table_post_process_config,
                       apply_image_post_processing, image_post_process_config,
                       api_key)
        elif os.path.isdir(input_path):
            process_directory(input_path, output_dir, config, use_direct_camelot,
                             apply_post_processing, post_process_config,
                             apply_table_post_processing, table_post_process_config,
                             apply_image_post_processing, image_post_process_config,
                             api_key)
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
                process_pdf(script_relative_path, output_dir, config, use_direct_camelot,
                           apply_post_processing, post_process_config,
                           apply_table_post_processing, table_post_process_config,
                           apply_image_post_processing, image_post_process_config,
                           api_key)
            elif os.path.isdir(script_relative_path):
                process_directory(script_relative_path, output_dir, config, use_direct_camelot,
                                 apply_post_processing, post_process_config,
                                 apply_table_post_processing, table_post_process_config,
                                 apply_image_post_processing, image_post_process_config,
                                 api_key)
            else:
                logger.error(f"Input path is neither a file nor a directory: {script_relative_path}")
                return 1
        else:
            # Path doesn't exist in any form we tried
            logger.error(f"Input path does not exist: {original_input_path}")
            logger.error(f"Also tried: {input_path} and {script_relative_path}")
            return 1
    
    logger.info("Processing complete")
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
