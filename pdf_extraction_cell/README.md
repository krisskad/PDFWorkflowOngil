# PDF Extraction Cell

A modular framework for extracting and organizing content from PDF documents.

## Features

- **Text Extraction**: Extract text content with page structure preservation
- **Metadata Extraction**: Extract document metadata (title, author, creation date, etc.)
- **Table Detection**: Identify and extract tabular data from PDFs
- **Image Detection**: Identify and extract images from PDFs
- **Flexible Configuration**: Configurable extraction settings via JSON
- **Multiple PDF Library Support**: Supports both PyMuPDF and PyPDF/PyPDF2 with fallback mechanisms

## Project Structure

```
pdf_extraction_cell/
├── data/
│   ├── input/     # Directory for input PDF files
│   └── output/    # Directory for extraction results
├── modules/
│   ├── pdf_parser.py            # Basic text and metadata extraction
│   ├── table_extractor.py       # Table extraction using Camelot and PDFPlumber
│   ├── image_extractor.py       # Image extraction using PyMuPDF and pdf2image
│   ├── vision_analyzer.py       # Vision API integration for complex visuals (planned)
│   └── content_organizer.py     # Organizes content into structured format (planned)
├── config/
│   └── extraction_config.json   # Configuration settings
├── utils/
│   ├── file_utils.py            # File operation utilities
│   └── image_utils.py           # Image processing utilities
├── tests/                       # For unit tests
├── main.py                      # Main script
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## Installation

1. Clone the repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Basic usage example:

```python
from pdf_extraction_cell.modules.pdf_parser import PDFParser
from pdf_extraction_cell.modules.table_extractor import TableExtractor
import json

# Load configuration
with open('config/extraction_config.json', 'r') as f:
    config = json.load(f)

# Initialize parser and table extractor
parser = PDFParser(config)
table_extractor = TableExtractor(config)

# Extract content from a PDF
pdf_path = 'data/input/sample.pdf'
result = parser.extract_all(pdf_path)

# Extract tables using the parser output
# Option 1: Use parser-guided table extraction (default)
tables = table_extractor.extract_tables_from_parser_output(pdf_path, result)
# Option 2: Use direct Camelot extraction without relying on parser-identified locations
# tables = table_extractor.extract_tables_from_parser_output(pdf_path, result, use_direct_camelot=True)
result["tables"] = tables

# Access extracted content
text_data = result['text']
metadata = result['metadata']
visual_elements = result['visual_elements']
tables_data = result['tables']

# Print some basic info
print(f"Document title: {metadata.get('title', 'Unknown')}")
print(f"Total pages: {metadata.get('page_count', 0)}")
print(f"Detected tables: {len(visual_elements.get('tables', []))}")
print(f"Extracted tables: {len(tables_data)}")
print(f"Detected images: {len(visual_elements.get('images', []))}")

# Convert tables to other formats if needed
markdown_tables = table_extractor.tables_to_markdown(tables_data)
html_tables = table_extractor.convert_tables_to_html(tables_data)
```

### Table Extractor

The `TableExtractor` class provides advanced table extraction capabilities:

- Extract tables from PDFs using multiple methods (Camelot, PDFPlumber)
- Use direct Camelot extraction or parser-guided extraction
- Post-process tables to improve quality
- Detect and clean up headers
- Merge similar tables across pages
- Convert tables to various formats (JSON, CSV, Markdown, HTML)
- Augment tables with contextual information from the PDF

Example of using specific table extraction features:

```python
# Option 1: Extract tables with parser-identified locations
tables = table_extractor.extract_tables(
    pdf_path,
    table_locations=result['visual_elements']['tables'],
    page_range="1-5"
)

# Option 2: Use direct Camelot extraction without relying on pre-identified locations
# tables = table_extractor.extract_tables(
#     pdf_path,
#     page_range="1-5",
#     use_direct_camelot=True
# )

# Merge similar tables that might be split across pages
merged_tables = table_extractor.merge_similar_tables(tables, similarity_threshold=0.8)

# Convert to markdown for documentation
markdown_tables = table_extractor.tables_to_markdown(merged_tables)
for page_num, tables_md in markdown_tables.items():
    print(f"Page {page_num} tables:")
    for table_md in tables_md:
        print(table_md)
```

### Image Extractor

The `ImageExtractor` class provides comprehensive image extraction capabilities:

- Extract images from PDFs using multiple methods (PyMuPDF, pdf2image)
- Automatic fallback between methods if one fails
- Extract vector graphics as SVG files
- Filter images based on size to avoid extracting small icons or artifacts
- Save extracted images in configurable formats (PNG, JPEG, etc.)
- Generate detailed metadata for each extracted image
- Extract images based on parser-identified locations or from specific pages

Example of using the image extractor:

```python
from pdf_extraction_cell.modules.pdf_parser import PDFParser
from pdf_extraction_cell.modules.image_extractor import ImageExtractor
import json

# Load configuration
with open('config/extraction_config.json', 'r') as f:
    config = json.load(f)

# Initialize parser and image extractor
parser = PDFParser(config)
image_extractor = ImageExtractor(config)

# Extract content from a PDF
pdf_path = 'data/input/sample.pdf'
result = parser.extract_all(pdf_path)

# Extract images using the parser output
images_output_dir = 'data/output/extracted_images'
images = image_extractor.extract_images_from_parser_output(pdf_path, result, output_dir=images_output_dir)

# Add images to the result
result["images"] = images

# Print image extraction summary
print(f"Extracted {len(images)} images from {pdf_path}")
for img in images:
    print(f"  - {img['filename']} ({img['width']}x{img['height']}) on page {img['page']}")

# Extract images from specific pages only
specific_pages = [1, 5, 10]
page_images = image_extractor.extract_images(
    pdf_path,
    page_range=specific_pages,
    output_dir=images_output_dir
)
```

## Command Line Usage

You can also use the command line interface to process PDF files:

```bash
# Process a single PDF file using parser-guided table extraction (default)
python main.py --input data/input/sample.pdf --output data/output

# Process a single PDF file using direct Camelot extraction
python main.py --input data/input/sample.pdf --output data/output --direct-camelot

# Process all PDFs in a directory
python main.py --input data/input --output data/output

# Process with a custom configuration file
python main.py --input data/input/sample.pdf --output data/output --config custom_config.json
```

## Configuration

The extraction process can be configured via the `extraction_config.json` file:

### Text Extraction Configuration
- `text_extraction_method`: Specify the PDF library to use ("pymupdf", "pypdf", or "auto")

### Table Extraction Configuration
- `table_extraction_method`: Specify the table extraction library ("camelot", "pdfplumber", or "auto")
- `table_flavor`: Table extraction flavor for Camelot ("lattice" or "stream")
- `line_scale`: Line scale parameter for Camelot (default: 15)
- `min_confidence`: Minimum confidence score for table extraction (0-100)
- `output_format`: Output format for tables ("json", "csv", or "both")
- `deduplicate_headers`: Whether to remove duplicate headers (true/false)
- `detect_headers`: Whether to auto-detect headers (true/false)
- `edge_tol`: Edge tolerance for Camelot (default: 50)
- `row_tol`: Row tolerance for Camelot (default: 2)

### Image Extraction Configuration
- `image_extraction_method`: Specify the image extraction library ("pymupdf", "pdf2image", or "auto")
- `min_image_size`: Minimum width/height in pixels for extracted images (default: 100)
- `image_output_format`: Output format for images ("png", "jpg", etc.)
- `image_output_dpi`: DPI for extracted images when using pdf2image (default: 300)
- `extract_vector_graphics`: Whether to extract vector graphics as SVG (true/false)
- `save_image_metadata`: Whether to save image metadata to JSON (true/false)

### General Configuration
- `extraction_settings`: Control which elements to extract
- `output_settings`: Configure output format and details
- `logging`: Configure logging behavior
- `performance`: Configure batch processing and parallelization

## Dependencies

- PyMuPDF (fitz) or PyPDF/PyPDF2 (at least one is required)
- Camelot and/or PDFPlumber (for table extraction)
- OpenCV and NumPy (for image processing)
- Pandas (for data manipulation)
- Pillow (for image handling)

## License

MIT
