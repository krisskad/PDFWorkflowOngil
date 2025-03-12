"""
Main entry point for the Chunking Pipeline.
This script processes PDF extraction outputs (text, tables, and charts) to generate unified JSON for RAG systems.
"""

import os
import sys
import logging
import argparse
import json
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import time

from chunking_cell.utils.helpers import load_config, read_json, write_json
from chunking_cell.modules.text_chunker import TextChunker
from chunking_cell.modules.table_chunker import TableChunker
from chunking_cell.modules.chart_chunker import ChartChunker
from chunking_cell.modules.relationship_detector import RelationshipDetector
from chunking_cell.modules.document_processor import DocumentProcessor
from chunking_cell.modules.json_assembler import JSONAssembler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("chunking_cell.main")

def process_document(
    document_prefix: str = None,
    text_path: str = None,
    tables_path: str = None,
    charts_path: str = None,
    output_path: str = None,
    config_path: str = "chunking_cell/config/chunking_config.json",
    document_id: str = None,
    company_name: str = None
) -> Dict[str, Any]:
    """
    Process a document through the chunking pipeline.
    
    Args:
        document_prefix: Prefix of the document files (e.g., "Apple_10K")
                        If provided, text_path, tables_path, and charts_path will be constructed automatically
        text_path: Path to the text JSON file (optional if document_prefix is provided)
        tables_path: Path to the tables JSON file (optional if document_prefix is provided)
        charts_path: Path to the charts JSON file (optional if document_prefix is provided)
        output_path: Path to save the output JSON file (optional if document_prefix is provided)
        config_path: Path to the configuration file
        document_id: Unique identifier for the document (optional, defaults to prefix if provided)
        company_name: Company name to include in chunk metadata (optional)
        
    Returns:
        Processed document as a dictionary
    """
    start_time = time.time()
    logger.info(f"Starting chunking pipeline for document")
    
    # Load configuration
    config = load_config(config_path)
    logger.info(f"Loaded configuration from {config_path}")
    
    # If document_prefix is provided, construct paths
    if document_prefix:
        # Get base paths from config
        text_base_path = config.get("input_paths", {}).get("text_json", "../pdf_extraction_cell/data/output/")
        tables_base_path = config.get("input_paths", {}).get("tables_json", "../pdf_extraction_cell/data/input/")
        charts_base_path = config.get("input_paths", {}).get("charts_json", "../pdf_extraction_cell/data/output/extracted_images/")
        output_base_path = config.get("output_path", "./data/output/")
        
        # Adjust paths to be relative to the project root
        # When running as a module, paths need to be relative to the current working directory
        if text_base_path.startswith("../"):
            text_base_path = text_base_path[3:]  # Remove "../" prefix
        if tables_base_path.startswith("../"):
            tables_base_path = tables_base_path[3:]  # Remove "../" prefix
        if charts_base_path.startswith("../"):
            charts_base_path = charts_base_path[3:]  # Remove "../" prefix
        
        # Construct file paths
        if not text_path:
            text_path = os.path.join(text_base_path, f"{document_prefix}_enhanced.json")
        if not tables_path:
            potential_tables_path = os.path.join(tables_base_path, f"{document_prefix}_tables_enhanced.json")
            tables_path = potential_tables_path if os.path.exists(potential_tables_path) else None
        if not charts_path:
            potential_charts_path = os.path.join(charts_base_path, f"{document_prefix}_charts.json")
            charts_path = potential_charts_path if os.path.exists(potential_charts_path) else None
        if not output_path:
            output_path = os.path.join(output_base_path, f"{document_prefix}_chunked.json")
    
    # Validate paths
    if not text_path or not output_path:
        raise ValueError("Either document_prefix or at least text_path and output_path must be provided")
    
    logger.info(f"Text path: {text_path}")
    logger.info(f"Tables path: {tables_path}")
    logger.info(f"Charts path: {charts_path}")
    logger.info(f"Output path: {output_path}")
    
    # Load configuration
    config = load_config(config_path)
    logger.info(f"Loaded configuration from {config_path}")
    
    # Read input files
    text_json = read_json(text_path)
    logger.info("Loaded text input file")
    
    # Handle optional inputs
    tables_json = read_json(tables_path) if tables_path and os.path.exists(tables_path) else None
    if tables_json:
        logger.info("Loaded tables input file")
    else:
        logger.info("Tables input file not provided or does not exist")
    
    charts_json = read_json(charts_path) if charts_path and os.path.exists(charts_path) else None
    if charts_json:
        logger.info("Loaded charts input file")
    else:
        logger.info("Charts input file not provided or does not exist")
    
    # Initialize processors
    text_chunker = TextChunker(config.get("text_chunking", {}))
    table_chunker = TableChunker(config.get("table_chunking", {}))
    chart_chunker = ChartChunker(config.get("chart_chunking", {}))
    relationship_detector = RelationshipDetector(config.get("relationship_detection", {}))
    document_processor = DocumentProcessor(config.get("document_processing", {}))
    json_assembler = JSONAssembler(config)
    logger.info("Initialized processors")
    
    # Use document_prefix as document_id if not provided
    if not document_id and document_prefix:
        document_id = document_prefix
    
    # Use document_id as file_id for chunk IDs
    file_id = document_id
    
    # Process inputs
    text_chunks = text_chunker.process(text_json, file_id, company_name)
    logger.info(f"Processed text chunks: {len(text_chunks)}")
    
    # Process tables if available
    table_chunks = []
    if tables_json:
        table_chunks = table_chunker.process(tables_json, file_id, company_name)
        logger.info(f"Processed table chunks: {len(table_chunks)}")
    else:
        logger.info("Skipping table chunk processing (no input)")
    
    # Process charts if available
    chart_chunks = []
    if charts_json:
        chart_chunks = chart_chunker.process(charts_json, file_id, company_name)
        logger.info(f"Processed chart chunks: {len(chart_chunks)}")
    else:
        logger.info("Skipping chart chunk processing (no input)")
    
    # Detect relationships
    relationships = relationship_detector.detect_relationships(
        text_chunks, table_chunks, chart_chunks
    )
    logger.info(f"Detected relationships: {len(relationships)}")
    
    # Extract document metadata
    metadata = document_processor.extract_metadata(
        text_json, tables_json or {}, charts_json or {}, document_id
    )
    logger.info(f"Extracted document metadata for document_id: {document_id}")
    
    # Assemble final JSON
    output = json_assembler.assemble(
        metadata, text_chunks, table_chunks, chart_chunks, relationships
    )
    logger.info("Assembled final JSON")
    
    # Write output to file
    write_json(output, output_path)
    logger.info(f"Wrote output to {output_path}")
    
    # Log processing time
    processing_time = time.time() - start_time
    logger.info(f"Completed chunking pipeline in {processing_time:.2f} seconds")
    
    return output

def main():
    """
    Main entry point for the chunking pipeline.
    """
    parser = argparse.ArgumentParser(description="Chunking Pipeline for PDF Extraction")
    
    # Create mutually exclusive group for document prefix or individual paths
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prefix", help="Document prefix (e.g., 'Apple_10K') to automatically find relevant files")
    group.add_argument("--text", help="Path to the text JSON file")
    
    # Other arguments
    parser.add_argument("--tables", help="Path to the tables JSON file (optional)")
    parser.add_argument("--charts", help="Path to the charts JSON file (optional)")
    parser.add_argument("--output", help="Path to save the output JSON file (optional)")
    parser.add_argument("--config", default="chunking_cell/config/chunking_config.json", help="Path to the configuration file")
    parser.add_argument("--document-id", help="Unique identifier for the document (optional, defaults to prefix if provided)")
    parser.add_argument("--company-name", help="Company name to include in chunk metadata (optional)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.prefix:
        # Use document prefix
        process_document(
            document_prefix=args.prefix,
            output_path=args.output,
            config_path=args.config,
            document_id=args.document_id,
            company_name=args.company_name
        )
    else:
        # Use individual paths
        process_document(
            text_path=args.text,
            tables_path=args.tables,
            charts_path=args.charts,
            output_path=args.output,
            config_path=args.config,
            document_id=args.document_id,
            company_name=args.company_name
        )

if __name__ == "__main__":
    main()
