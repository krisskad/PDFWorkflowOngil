# Online Augmentation

A Python package for extracting specific entities from chunked JSON files.

## Overview

The Online Augmentation package provides functionality to extract specific entities and metadata from chunked JSON files. It allows you to configure which entity types to extract, set confidence thresholds, and control the output format.

## Installation

No installation is required. The package can be run directly from the source code.

## Usage

### Command Line Interface

The package provides a command-line interface for extracting entities from chunked JSON files:

```bash
# Process a single file
python online_augmentation/main.py --input data/output/gazette_sample_chunked.json

# Process a single file with a custom output path
python online_augmentation/main.py --input data/output/gazette_sample_chunked.json --output custom_output.json

# Process a single file with a custom configuration
python online_augmentation/main.py --input data/output/gazette_sample_chunked.json --config path/to/config.json

# Process all files matching the pattern in the configuration
python online_augmentation/main.py --batch
```

### Python API

You can also use the package as a Python API:

```python
from online_augmentation.main import process_file

# Process a single file
result = process_file(
    input_file_path="data/output/gazette_sample_chunked.json",
    output_file_path="data/output/augmented/gazette_sample_augmented.json",
    config_path="online_augmentation/config/augmentation_config.json"
)
```

## Configuration

The package uses a JSON configuration file to control the extraction process. The default configuration file is located at `online_augmentation/config/augmentation_config.json`.

### Configuration Options

```json
{
  "input": {
    "file_path": "",
    "default_input_dir": "./data/output/",
    "file_pattern": "*_chunked.json"
  },
  "output": {
    "output_dir": "./data/output/augmented/",
    "file_name_suffix": "_augmented"
  },
  "entity_extraction": {
    "enabled": true,
    "entity_types": [
      "Organizations",
      "People",
      "Locations",
      "Dates",
      "Financial figures",
      "Products or services",
      "Legal references",
      "Road_Sections"
    ],
    "confidence_threshold": 0.7,
    "include_context": true,
    "max_entities_per_type": 100
  },
  "metadata_extraction": {
    "enabled": true,
    "fields": [
      "document_id",
      "file_info",
      "document_type",
      "summary",
      "topics",
      "key_information"
    ]
  },
  "processing": {
    "batch_size": 10,
    "max_workers": 4,
    "log_level": "INFO"
  }
}
```

### Configuration Fields

#### Input Configuration

- `file_path`: Path to a specific input file. If provided, this file will be processed instead of searching for files matching the pattern.
- `default_input_dir`: Directory to search for input files if `file_path` is not provided.
- `file_pattern`: Pattern to match input files in the `default_input_dir`.

#### Output Configuration

- `output_dir`: Directory to save output files.
- `file_name_suffix`: Suffix to add to the input file name to create the output file name.

#### Entity Extraction Configuration

- `enabled`: Whether to extract entities.
- `entity_types`: List of entity types to extract.
- `confidence_threshold`: Minimum confidence threshold for entities to be included.
- `include_context`: Whether to include the context field in the extracted entities.
- `max_entities_per_type`: Maximum number of entities to include for each entity type.

#### Metadata Extraction Configuration

- `enabled`: Whether to extract metadata.
- `fields`: List of metadata fields to extract.

#### Processing Configuration

- `batch_size`: Number of files to process in a batch.
- `max_workers`: Number of worker threads to use for processing.
- `log_level`: Logging level.

## Command-Line Arguments

- `--input`: Path to the input file.
- `--output`: Path to save the output file (optional).
- `--config`: Path to the configuration file (optional, defaults to `online_augmentation/config/augmentation_config.json`).
- `--batch`: Process all files matching the pattern in the configuration.

## Output Format

The output file will contain the extracted entities and metadata in JSON format:

```json
{
  "metadata": {
    "document_id": "...",
    "file_info": { ... },
    "document_type": { ... },
    "summary": { ... },
    "topics": { ... },
    "key_information": { ... }
  },
  "entities": {
    "Organizations": [ ... ],
    "People": [ ... ],
    "Locations": [ ... ],
    "Dates": [ ... ],
    "Financial figures": [ ... ],
    "Products or services": [ ... ],
    "Legal references": [ ... ],
    "Road_Sections": [ ... ]
  },
  "processing_info": {
    "timestamp": 1678654321.123,
    "input_file": "data/output/gazette_sample_chunked.json",
    "config_file": "online_augmentation/config/augmentation_config.json"
  }
}
