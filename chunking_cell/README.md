# PDF Chunking Pipeline

This module processes PDF extraction outputs (text, tables, and charts) to generate unified JSON for RAG (Retrieval-Augmented Generation) systems. The pipeline integrates content into discrete, searchable chunks while preserving document structure and relationships between different content types.

## Overview

The chunking pipeline takes the enhanced outputs from the PDF extraction cell and creates a unified JSON structure optimized for RAG systems. The pipeline:

1. Processes text content using sliding window chunking
2. Processes tables as individual chunks
3. Processes charts as individual chunks
4. Detects relationships between chunks
5. Extracts document-level metadata
6. Assembles everything into a unified JSON structure

## Input Files

The pipeline expects three separate JSON files per document:

- **Text JSON**: Contains extracted text with page-level AI enhancements (from `pdf_extraction_cell/data/output/`)
- **Tables JSON**: Contains extracted tables with metadata and AI enhancements (from `pdf_extraction_cell/data/input/`)
- **Charts JSON**: Contains processed images/charts with AI enhancements (from `pdf_extraction_cell/data/output/extracted_images/`)

## Output Structure

The pipeline generates a unified JSON structure with:

- Document-level metadata
- Array of discrete chunks (text, tables, charts)
- Relationships between chunks
- Document structure (sections, hierarchy)
- Processing metadata

### Chunk Types

#### Text Chunks

- Unique ID following pattern `text_p{page_number}_{chunk_index}`
- Page number, section context
- Chunk text content
- Topics and entities from AI enhancements
- Window position for tracking overlap

#### Table Chunks

- Unique ID following pattern `table_p{page_number}_{table_index}`
- Table data (headers, rows)
- AI enhancements (summary, insights, keywords)
- Relevant metadata (title, footnotes, etc.)

#### Chart Chunks

- Unique ID following pattern `chart_p{page_number}_{chart_index}`
- Chart metadata (type, description)
- Extracted tabular data
- AI enhancements (summary, trends, insights)
- Image path reference

### Relationships

The pipeline detects and stores relationships between chunks:

- Text chunks that reference tables or charts
- Related data between tables and charts
- Adjacent content in the same section

## Usage

### Command Line

You can use the chunking pipeline in two ways:

#### 1. Using Document Prefix (Recommended)

Simply provide a document prefix, and the pipeline will automatically locate the relevant files:

```bash
python -m chunking_cell.main --prefix Apple_10K
```

This will automatically find:
- `../pdf_extraction_cell/data/output/Apple_10K_enhanced.json` (text)
- `../pdf_extraction_cell/data/input/Apple_10K_tables_enhanced.json` (tables)
- `../pdf_extraction_cell/data/output/extracted_images/Apple_10K_charts.json` (charts)

And output to:
- `./data/output/Apple_10K_chunked.json`

#### 2. Using Individual File Paths

Alternatively, you can specify each file path individually:

```bash
python -m chunking_cell.main --text path/to/text.json --tables path/to/tables.json --charts path/to/charts.json --output path/to/output.json
```

### Python API

Similarly, the Python API supports both approaches:

#### 1. Using Document Prefix

```python
from chunking_cell.main import process_document

output = process_document(
    document_prefix="Apple_10K",
    config_path="path/to/config.json"  # Optional
)
```

#### 2. Using Individual File Paths

```python
from chunking_cell.main import process_document

output = process_document(
    text_path="path/to/text.json",
    tables_path="path/to/tables.json",
    charts_path="path/to/charts.json",
    output_path="path/to/output.json",
    config_path="path/to/config.json"  # Optional
)
```

## Configuration

The pipeline can be configured using a JSON configuration file. The default configuration is in `config/chunking_config.json`.

```json
{
  "input_paths": {
    "text_json": "../pdf_extraction_cell/data/output/",
    "tables_json": "../pdf_extraction_cell/data/input/",
    "charts_json": "../pdf_extraction_cell/data/output/extracted_images/"
  },
  "output_path": "./data/output/",
  
  "text_chunking": {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "min_chunk_size": 100
  },
  
  "table_chunking": {
    "include_summary": true,
    "include_insights": true
  },
  
  "chart_chunking": {
    "include_summary": true,
    "include_trends": true,
    "include_insights": true
  },
  
  "relationship_detection": {
    "table_reference_patterns": ["Table", "table", "tbl"],
    "chart_reference_patterns": ["Figure", "figure", "fig", "chart", "graph"],
    "max_distance_for_adjacency": 3
  },
  
  "document_processing": {
    "include_summary": true,
    "include_topics": true,
    "include_entities": true,
    "include_structure": true
  }
}
```

## Installation

1. Clone the repository
2. Install the requirements:

```bash
pip install -r chunking_cell/requirements.txt
```

## Dependencies

- Python 3.8+
- Dependencies listed in `requirements.txt`

## Module Structure

- `chunking_cell/`: Main package
  - `modules/`: Core modules
    - `text_chunker.py`: Processes text content
    - `table_chunker.py`: Processes table content
    - `chart_chunker.py`: Processes chart content
    - `relationship_detector.py`: Detects relationships between chunks
    - `document_processor.py`: Extracts document metadata
    - `json_assembler.py`: Assembles final JSON output
  - `utils/`: Utility functions
    - `helpers.py`: Helper functions
  - `config/`: Configuration files
    - `chunking_config.json`: Configuration for the chunking pipeline
  - `main.py`: Main entry point
  - `requirements.txt`: Dependencies
  - `README.md`: Documentation
