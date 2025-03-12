"""
Enhanced Table Extraction Module for extracting tables from PDFs.
"""
import os
import logging
import json
import csv
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple

# Try importing table extraction libraries with fallbacks
try:
    import camelot
except ImportError:
    camelot = None
try:
    import pdfplumber
except ImportError:
    pdfplumber = None

logger = logging.getLogger("pdf_extraction.table_extractor")

class TableExtractor:
    """Class for extracting tables from PDF documents using various methods."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the table extractor with configuration.
        
        Args:
            config: Configuration dictionary for the extractor
        """
        self.config = config
        self.extraction_method = config.get("table_extraction_method", "auto")
        self.flavor = config.get("table_flavor", "lattice")  # lattice or stream
        self.table_areas = config.get("table_areas", None)  # Pre-defined table areas
        self.line_scale = config.get("line_scale", 15)  # For camelot-py
        self.min_confidence = config.get("min_confidence", 70)  # Minimum confidence score (0-100)
        self.output_format = config.get("output_format", "json")  # json, csv, or both
        self.deduplicate_headers = config.get("deduplicate_headers", True)  # Remove duplicate headers
        self.detect_headers = config.get("detect_headers", True)  # Auto-detect headers
        self.edge_tol = config.get("edge_tol", 50)  # Edge tolerance for camelot
        self.row_tol = config.get("row_tol", 2)  # Row tolerance for camelot
        
        # Table filtering options
        self.filter_non_tabular = config.get("filter_non_tabular", True)
        self.min_table_rows = config.get("min_table_rows", 2)
        self.min_table_cols = config.get("min_table_cols", 2)
        self.min_cell_density = config.get("min_cell_density", 0.3)
        self.max_column_variation = config.get("max_column_variation", 2)
        
        self.validate_dependencies()
        
        logger.info(f"Initialized Table Extractor with method: {self.extraction_method}")
    
    def validate_dependencies(self) -> None:
        """Validate that required dependencies are available based on the chosen method."""
        if self.extraction_method == "camelot" and camelot is None:
            logger.warning("Camelot is not installed. Falling back to another method.")
            if pdfplumber is not None:
                self.extraction_method = "pdfplumber"
            else:
                logger.error("No table extraction libraries available. Tables cannot be extracted.")
                self.extraction_method = "none"
                
        elif self.extraction_method == "pdfplumber" and pdfplumber is None:
            logger.warning("PDFPlumber is not installed. Falling back to another method.")
            if camelot is not None:
                self.extraction_method = "camelot"
            else:
                logger.error("No table extraction libraries available. Tables cannot be extracted.")
                self.extraction_method = "none"
                
        elif self.extraction_method == "auto":
            # Choose the best available method (prefer camelot over pdfplumber)
            if camelot is not None:
                self.extraction_method = "camelot"
            elif pdfplumber is not None:
                self.extraction_method = "pdfplumber"
            else:
                logger.error("No table extraction libraries available. Tables cannot be extracted.")
                self.extraction_method = "none"
    
    def extract_tables(self, pdf_path: str, 
                       table_locations: Optional[List[Dict[str, Any]]] = None, 
                       page_range: Optional[Union[str, List[int]]] = None,
                       use_direct_camelot: bool = False) -> List[Dict[str, Any]]:
        """
        Extract tables from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            table_locations: Optional list of potential table locations identified by the PDF Parser
            page_range: Optional page range to extract tables from
            use_direct_camelot: If True, use Camelot directly for table detection without relying on pre-identified locations
            
        Returns:
            List of dictionaries containing extracted table data
        """
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file does not exist: {pdf_path}")
            return []
            
        logger.info(f"Extracting tables from {pdf_path} using {self.extraction_method}")
        
        if self.extraction_method == "none":
            logger.warning("No table extraction method available. Returning empty result.")
            return []
        
        # Determine pages to process
        if page_range is None and table_locations:
            # Extract tables only from pages where potential tables were identified
            pages = sorted(set(loc["page"] for loc in table_locations if "page" in loc))
            if not pages:
                page_range = "all"
            else:
                page_range = ",".join(str(p) for p in pages)
        
        # Extract tables using the selected method
        if use_direct_camelot and camelot is not None:
            tables_data = self._extract_tables_direct_camelot(pdf_path, page_range)
        elif self.extraction_method == "camelot":
            tables_data = self._extract_tables_camelot(pdf_path, table_locations, page_range)
        elif self.extraction_method == "pdfplumber":
            tables_data = self._extract_tables_pdfplumber(pdf_path, table_locations, page_range)
        else:
            logger.error(f"Unsupported extraction method: {self.extraction_method}")
            return []
            
        # Post-process the extracted tables
        processed_tables = self.post_process_tables(tables_data)
        
        # Save to JSON if requested
        if self.output_format in ["json", "both"]:
            output_dir = os.path.dirname(pdf_path)
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            json_path = os.path.join(output_dir, f"{base_name}_tables.json")
            self.save_tables_to_json(processed_tables, json_path)
        
        return processed_tables
    
    def _extract_tables_direct_camelot(self, pdf_path: str, 
                                      page_range: Optional[Union[str, List[int]]]) -> List[Dict[str, Any]]:
        """
        Extract tables using Camelot directly without relying on pre-identified table locations.
        Camelot has its own table detection capabilities.
        
        Args:
            pdf_path: Path to the PDF file
            page_range: Optional page range to extract tables from
            
        Returns:
            List of dictionaries containing extracted table data
        """
        tables_data = []
        
        try:
            # Prepare page ranges
            pages = page_range if page_range else "all"
            
            # Try both flavors of Camelot (lattice and stream) for best results
            flavors = ["lattice", "stream"]
            all_tables = []
            
            for flavor in flavors:
                logger.info(f"Attempting table extraction with Camelot using {flavor} flavor")
                
                # Prepare kwargs based on flavor
                kwargs = {"pages": pages}
                
                if flavor == "stream":
                    kwargs["edge_tol"] = self.edge_tol
                    kwargs["row_tol"] = self.row_tol
                else:  # lattice flavor
                    kwargs["line_scale"] = self.line_scale
                    kwargs["process_background"] = True
                
                try:
                    # Extract tables with current flavor
                    tables = camelot.read_pdf(pdf_path, flavor=flavor, **kwargs)
                    
                    if len(tables) > 0:
                        logger.info(f"Found {len(tables)} tables using Camelot with {flavor} flavor")
                        
                        # Add flavor to each table for tracking
                        for table in tables:
                            table.flavor = flavor
                        
                        all_tables.extend(tables)
                except Exception as e:
                    logger.warning(f"Error extracting tables with Camelot {flavor} flavor: {str(e)}")
            
            # If no tables found with either flavor, try PDFPlumber as a fallback
            if len(all_tables) == 0 and pdfplumber is not None:
                logger.info("No tables found with Camelot, falling back to PDFPlumber")
                return self._extract_tables_pdfplumber(pdf_path, None, page_range)
            
            # Process extracted tables
            for i, table in enumerate(all_tables):
                # Skip low-quality tables
                if table.accuracy < self.min_confidence:
                    logger.warning(f"Skipping table {i+1} due to low confidence: {table.accuracy}")
                    continue
                
                # Convert table to structured data
                table_data = {
                    "page": table.page,
                    "table_number": i + 1,
                    "data": table.data,
                    "metadata": {
                        "accuracy": table.accuracy,
                        "extraction_method": "camelot_direct",
                        "flavor": table.flavor if hasattr(table, 'flavor') else self.flavor,
                        "whitespace": table.whitespace,
                        "headers": table.headers if hasattr(table, 'headers') else [],
                        "shape": table.shape
                    }
                }
                
                tables_data.append(table_data)
                
                # Save to CSV if requested
                if self.output_format in ["csv", "both"]:
                    output_dir = os.path.dirname(pdf_path)
                    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
                    csv_path = os.path.join(output_dir, f"{base_name}_table_{table.page}_{i+1}.csv")
                    table.to_csv(csv_path)
                    table_data["metadata"]["csv_path"] = csv_path
            
            logger.info(f"Successfully extracted {len(tables_data)} tables from {pdf_path} using direct Camelot")
            
        except Exception as e:
            logger.error(f"Error extracting tables with direct Camelot: {str(e)}")
            # Try fallback if failed
            if pdfplumber is not None:
                logger.info("Falling back to PDFPlumber for table extraction")
                return self._extract_tables_pdfplumber(pdf_path, None, page_range)
            else:
                return []
        
        return tables_data
    
    def _extract_tables_camelot(self, pdf_path: str, 
                               table_locations: Optional[List[Dict[str, Any]]], 
                               page_range: Optional[Union[str, List[int]]]) -> List[Dict[str, Any]]:
        """Extract tables using Camelot with pre-identified table locations."""
        tables_data = []
        
        try:
            # Prepare page ranges
            pages = page_range if page_range else "all"
            
            # Prepare table areas if provided
            table_areas = []
            if table_locations:
                for loc in table_locations:
                    if "rect" in loc and loc["rect"]:
                        # Convert rect to Camelot format [x1, y1, x2, y2]
                        # Ensure rect is a list, not a tuple
                        rect = loc["rect"]
                        if rect is None:
                            continue  # Skip None values
                        if isinstance(rect, tuple):
                            rect = list(rect)
                        # Ensure all elements in rect are valid numbers
                        if len(rect) == 4 and all(isinstance(val, (int, float)) for val in rect):
                            table_areas.append(rect)
                        else:
                            logger.warning(f"Skipping invalid rect: {rect}")
            
            # Prepare base kwargs
            kwargs = {"pages": pages}
            
            # Add flavor-specific parameters
            if self.flavor == "stream":
                kwargs["edge_tol"] = self.edge_tol
                kwargs["row_tol"] = self.row_tol
                # Add table areas if available
                if table_areas:
                    try:
                        # Camelot expects table_areas as a list of strings, each formatted as "x1,y1,x2,y2"
                        formatted_areas = [','.join(map(str, area)) for area in table_areas]
                        kwargs["table_areas"] = formatted_areas
                    except Exception as e:
                        logger.warning(f"Error formatting table areas: {str(e)}")
                        # Continue without table areas
            else:  # lattice flavor
                kwargs["line_scale"] = self.line_scale
                kwargs["process_background"] = True
            
            # Extract tables
            tables = camelot.read_pdf(pdf_path, flavor=self.flavor, **kwargs)
            
            logger.info(f"Found {len(tables)} tables using Camelot")
            
            # If no tables found with current flavor, try the other flavor
            if len(tables) == 0:
                other_flavor = "stream" if self.flavor == "lattice" else "lattice"
                logger.info(f"No tables found with {self.flavor} flavor, trying {other_flavor} flavor")
                
                # Prepare kwargs for the other flavor
                other_kwargs = {"pages": pages}
                if other_flavor == "stream":
                    other_kwargs["edge_tol"] = self.edge_tol
                    other_kwargs["row_tol"] = self.row_tol
                    if table_areas:
                        try:
                            # Camelot expects table_areas as a list of strings, each formatted as "x1,y1,x2,y2"
                            formatted_areas = [','.join(map(str, area)) for area in table_areas]
                            other_kwargs["table_areas"] = formatted_areas
                        except Exception as e:
                            logger.warning(f"Error formatting table areas for other flavor: {str(e)}")
                            # Continue without table areas
                else:  # lattice flavor
                    other_kwargs["line_scale"] = self.line_scale
                    other_kwargs["process_background"] = True
                
                try:
                    tables = camelot.read_pdf(pdf_path, flavor=other_flavor, **other_kwargs)
                    logger.info(f"Found {len(tables)} tables using Camelot with {other_flavor} flavor")
                except Exception as e:
                    logger.error(f"Error extracting tables with Camelot {other_flavor} flavor: {str(e)}")
            
            # If still no tables found, try PDFPlumber as a fallback
            if len(tables) == 0 and pdfplumber is not None:
                logger.info("No tables found with Camelot, falling back to PDFPlumber")
                return self._extract_tables_pdfplumber(pdf_path, table_locations, page_range)
            
            # Process extracted tables
            for i, table in enumerate(tables):
                # Skip low-quality tables
                if table.accuracy < self.min_confidence:
                    logger.warning(f"Skipping table {i+1} due to low confidence: {table.accuracy}")
                    continue
                
                # Convert table to structured data
                table_data = {
                    "page": table.page,
                    "table_number": i + 1,
                    "data": table.data,
                    "metadata": {
                        "accuracy": table.accuracy,
                        "extraction_method": "camelot",
                        "flavor": self.flavor,
                        "whitespace": table.whitespace,
                        "headers": table.headers if hasattr(table, 'headers') else [],
                        "shape": table.shape
                    }
                }
                
                tables_data.append(table_data)
                
                # Save to CSV if requested
                if self.output_format in ["csv", "both"]:
                    output_dir = os.path.dirname(pdf_path)
                    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
                    csv_path = os.path.join(output_dir, f"{base_name}_table_{table.page}_{i+1}.csv")
                    table.to_csv(csv_path)
                    table_data["metadata"]["csv_path"] = csv_path
            
            logger.info(f"Successfully extracted {len(tables_data)} tables from {pdf_path} using Camelot")
            
        except Exception as e:
            logger.error(f"Error extracting tables with Camelot: {str(e)}")
            # Try fallback if failed
            if pdfplumber is not None:
                logger.info("Falling back to PDFPlumber for table extraction")
                return self._extract_tables_pdfplumber(pdf_path, table_locations, page_range)
            else:
                return []
        
        return tables_data
    
    def _extract_tables_pdfplumber(self, pdf_path: str, 
                                  table_locations: Optional[List[Dict[str, Any]]], 
                                  page_range: Optional[Union[str, List[int]]]) -> List[Dict[str, Any]]:
        """Extract tables using PDFPlumber."""
        tables_data = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                # Determine which pages to process
                if isinstance(page_range, str) and page_range == "all":
                    pages_to_process = range(len(pdf.pages))
                elif isinstance(page_range, str):
                    # Convert string like "1-3,5,7-9" to a list of integers
                    pages_to_process = []
                    for part in page_range.split(','):
                        if '-' in part:
                            start, end = map(int, part.split('-'))
                            pages_to_process.extend(range(start - 1, end))  # 0-indexed
                        else:
                            pages_to_process.append(int(part) - 1)  # 0-indexed
                elif isinstance(page_range, list):
                    pages_to_process = [p - 1 for p in page_range]  # 0-indexed
                else:
                    pages_to_process = range(len(pdf.pages))
                
                table_count = 0
                
                # Process each page
                for page_idx in pages_to_process:
                    if page_idx >= len(pdf.pages):
                        logger.warning(f"Page {page_idx+1} out of range, skipping")
                        continue
                    
                    page = pdf.pages[page_idx]
                    
                    # Try to use table_locations if provided
                    custom_settings = {}
                    if table_locations:
                        page_locations = [loc for loc in table_locations 
                                         if "page" in loc and loc["page"] == page_idx + 1]
                        
                        # PDFPlumber's extract_tables() doesn't accept these parameters
                        # Use an empty dict for now - we'll rely on default settings
                        if page_locations:
                            custom_settings = {}
                    
                    # Extract tables from the page
                    tables = page.extract_tables(**custom_settings)
                    
                    # If no tables found with default settings, try without any settings
                    # PDFPlumber's extract_tables() doesn't accept these parameters
                    if not tables:
                        tables = page.extract_tables()
                    
                    # Process found tables
                    for i, table in enumerate(tables):
                        # Skip empty tables
                        if not table or all(not row or all(cell == "" for cell in row) for row in table):
                            logger.warning(f"Skipping empty table {i+1} on page {page_idx+1}")
                            continue
                        
                        table_count += 1
                        
                        # Clean table data (remove None values)
                        cleaned_table = []
                        for row in table:
                            cleaned_row = [str(cell).strip() if cell is not None else "" for cell in row]
                            cleaned_table.append(cleaned_row)
                        
                        # Convert table to structured data
                        table_data = {
                            "page": page_idx + 1,  # 1-indexed
                            "table_number": table_count,
                            "data": cleaned_table,
                            "metadata": {
                                "extraction_method": "pdfplumber",
                                "shape": (len(cleaned_table), len(cleaned_table[0]) if cleaned_table else 0)
                            }
                        }
                        
                        tables_data.append(table_data)
                        
                        # Save to CSV if requested
                        if self.output_format in ["csv", "both"]:
                            output_dir = os.path.dirname(pdf_path)
                            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
                            csv_path = os.path.join(output_dir, f"{base_name}_table_{page_idx+1}_{i+1}.csv")
                            
                            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                                writer = csv.writer(f)
                                writer.writerows(cleaned_table)
                            
                            table_data["metadata"]["csv_path"] = csv_path
                
                logger.info(f"Successfully extracted {table_count} tables from {pdf_path} using PDFPlumber")
                
        except Exception as e:
            logger.error(f"Error extracting tables with PDFPlumber: {str(e)}")
        
        return tables_data
    
    def post_process_tables(self, tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Post-process extracted tables to improve quality.
        
        Args:
            tables: List of dictionaries containing table data
            
        Returns:
            List of dictionaries containing post-processed table data
        """
        processed_tables = []
        
        for table in tables:
            # Copy table to avoid modifying the original
            processed_table = table.copy()
            data = processed_table["data"]
            
            # Skip empty tables
            if not data or len(data) == 0:
                logger.warning(f"Skipping empty table {table.get('table_number', 'unknown')} on page {table.get('page', 'unknown')}")
                continue
                
            # Handle empty cells
            for i, row in enumerate(data):
                for j, cell in enumerate(row):
                    if cell is None or cell == "":
                        data[i][j] = ""
            
            # Remove completely empty rows
            data = [row for row in data if any(cell.strip() for cell in row)]
            
            if not data:  # Skip if all rows were empty
                logger.warning(f"Skipping table with only empty rows on page {table.get('page', 'unknown')}")
                continue
                
            # Try to detect and remove duplicate headers if enabled
            if self.deduplicate_headers and len(data) >= 2:
                potential_header = data[0]
                next_row = data[1]
                
                # Check if the next row contains similar content to the header
                similar_count = sum(1 for h, c in zip(potential_header, next_row) 
                                  if h and c and (h.lower() == c.lower() or h.lower() in c.lower()))
                
                if similar_count >= len(potential_header) * 0.5:
                    logger.info(f"Detected and removing duplicate header in table {table['table_number']} on page {table['page']}")
                    data = data[1:]  # Remove duplicate header
            
            # Try to detect and set header if none exists and detection is enabled
            if self.detect_headers and ("headers" not in processed_table["metadata"] or not processed_table["metadata"]["headers"]):
                if data and all(isinstance(cell, str) and cell for cell in data[0]):
                    # Check if first row looks like a header (all cells non-empty and contain text)
                    processed_table["metadata"]["headers"] = data[0]
                    processed_table["metadata"]["has_detected_header"] = True
                    # Remove the header row from data to avoid duplication
                    data = data[1:]
            
            # Clean headers if present
            if "headers" in processed_table["metadata"] and processed_table["metadata"]["headers"]:
                headers = processed_table["metadata"]["headers"]
                # Remove extra whitespace and newlines from headers
                processed_table["metadata"]["headers"] = [
                    h.strip().replace('\n', ' ').replace('  ', ' ') if isinstance(h, str) else h 
                    for h in headers
                ]
            
            # Standardize row lengths
            max_cols = max(len(row) for row in data) if data else 0
            for i, row in enumerate(data):
                if len(row) < max_cols:
                    data[i] = row + [""] * (max_cols - len(row))
            
            # Update processed data
            processed_table["data"] = data
            processed_table["metadata"]["shape"] = (len(data), max_cols if data else 0)
            processed_tables.append(processed_table)
        
        return processed_tables
    
    def save_tables_to_json(self, tables: List[Dict[str, Any]], output_path: str) -> None:
        """
        Save extracted tables to a JSON file.
        
        Args:
            tables: List of dictionaries containing table data
            output_path: Path to save the JSON file
        """
        try:
            # Ensure the output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(tables, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(tables)} tables to {output_path}")
        except Exception as e:
            logger.error(f"Error saving tables to JSON: {str(e)}")
    
    def tables_to_markdown(self, tables: List[Dict[str, Any]]) -> Dict[int, List[str]]:
        """
        Convert extracted tables to Markdown format.
        
        Args:
            tables: List of dictionaries containing table data
            
        Returns:
            Dictionary mapping page numbers to lists of Markdown tables
        """
        markdown_tables = {}
        
        for table in tables:
            page = table['page']
            data = table['data']
            
            if not data:
                continue
            
            # Initialize list for the page if it doesn't exist
            if page not in markdown_tables:
                markdown_tables[page] = []
            
            # Create Markdown table
            md_lines = []
            
            # Add title if available
            table_num = table['table_number']
            md_lines.append(f"**Table {table_num} (Page {page})**\n")
            
            # Add headers if available
            headers = table['metadata'].get('headers', [])
            
            # If no explicit headers but data exists, use first row as header
            if not headers and data:
                headers = data[0]
                data = data[1:]  # Remove header row from data
            
            # Create Markdown table
            if headers:
                # Format headers
                header_row = "| " + " | ".join(str(h) if h is not None else "" for h in headers) + " |"
                md_lines.append(header_row)
                
                # Add separator row
                separator = "| " + " | ".join(["---"] * len(headers)) + " |"
                md_lines.append(separator)
            
            # Add data rows
            for row in data:
                # Skip rows with all empty cells
                if all(not cell for cell in row):
                    continue
                    
                # Format row, handling potential missing columns
                row_str = "| " + " | ".join(str(cell) if cell is not None else "" for cell in row)
                
                # Add missing columns if necessary
                if headers and len(row) < len(headers):
                    row_str += " | " * (len(headers) - len(row))
                
                row_str += " |"
                md_lines.append(row_str)
            
            # Add empty line after table
            md_lines.append("")
            
            # Add table metadata as a comment
            metadata = {k: v for k, v in table['metadata'].items() if k not in ['csv_path', 'headers']}
            md_lines.append(f"<!-- Table metadata: {json.dumps(metadata)} -->\n")
            
            # Add to the page's tables
            markdown_tables[page].append("\n".join(md_lines))
        
        return markdown_tables
    
    def convert_tables_to_html(self, tables: List[Dict[str, Any]]) -> Dict[int, List[str]]:
        """
        Convert extracted tables to HTML format.
        
        Args:
            tables: List of dictionaries containing table data
            
        Returns:
            Dictionary mapping page numbers to lists of HTML tables
        """
        html_tables = {}
        
        for table in tables:
            page = table['page']
            data = table['data']
            
            if not data:
                continue
            
            # Initialize list for the page if it doesn't exist
            if page not in html_tables:
                html_tables[page] = []
            
            # Create HTML table
            html_lines = []
            
            # Add title
            table_num = table['table_number']
            html_lines.append(f'<div class="table-container">')
            html_lines.append(f'<h3>Table {table_num} (Page {page})</h3>')
            
            # Start table
            html_lines.append('<table border="1" cellpadding="3" cellspacing="0">')
            
            # Add headers if available
            headers = table['metadata'].get('headers', [])
            
            # If no explicit headers but data exists, use first row as header
            if not headers and data:
                headers = data[0]
                data = data[1:]  # Remove header row from data
            
            # Create HTML table header
            if headers:
                html_lines.append('<thead>')
                html_lines.append('<tr>')
                for header in headers:
                    html_lines.append(f'<th>{header}</th>')
                html_lines.append('</tr>')
                html_lines.append('</thead>')
            
            # Create HTML table body
            html_lines.append('<tbody>')
            for row in data:
                # Skip rows with all empty cells
                if all(not cell for cell in row):
                    continue
                
                html_lines.append('<tr>')
                for cell in row:
                    html_lines.append(f'<td>{cell}</td>')
                html_lines.append('</tr>')
            html_lines.append('</tbody>')
            
            # End table
            html_lines.append('</table>')
            
            # Add table metadata as a comment
            metadata = {k: v for k, v in table['metadata'].items() if k not in ['csv_path', 'headers']}
            html_lines.append(f'<!-- Table metadata: {json.dumps(metadata)} -->')
            html_lines.append('</div>')
            
            # Add to the page's tables
            html_tables[page].append("\n".join(html_lines))
        
        return html_tables
    
    def merge_similar_tables(self, tables: List[Dict[str, Any]], 
                            similarity_threshold: float = 0.8) -> List[Dict[str, Any]]:
        """
        Merge tables that appear to be parts of the same logical table (e.g., continued tables).
        
        Args:
            tables: List of dictionaries containing table data
            similarity_threshold: Threshold for considering tables similar (0.0-1.0)
            
        Returns:
            List of merged tables
        """
        if not tables or len(tables) <= 1:
            return tables
            
        # Sort tables by page number and position
        sorted_tables = sorted(tables, key=lambda t: (t['page'], t.get('metadata', {}).get('order', 0)))
        merged_tables = []
        current_table = sorted_tables[0]
        
        for next_table in sorted_tables[1:]:
            # Check if tables are on consecutive pages or same page
            consecutive_pages = (next_table['page'] == current_table['page'] + 1)
            same_page = (next_table['page'] == current_table['page'])
            
            # Check if tables have similar structure
            current_cols = len(current_table['data'][0]) if current_table['data'] else 0
            next_cols = len(next_table['data'][0]) if next_table['data'] else 0
            similar_structure = (current_cols == next_cols) or (
                abs(current_cols - next_cols) / max(current_cols, next_cols) < (1 - similarity_threshold)
            )
            
            # Check headers similarity if both have headers
            headers_match = False
            if ('headers' in current_table.get('metadata', {}) and 
                'headers' in next_table.get('metadata', {})):
                current_headers = current_table['metadata']['headers']
                next_headers = next_table['metadata']['headers']
                
                if current_headers and next_headers:
                    # Count matching headers
                    matches = sum(1 for h1, h2 in zip(current_headers, next_headers) 
                                if h1 and h2 and h1.lower() == h2.lower())
                    total = max(len(current_headers), len(next_headers))
                    headers_match = matches / total >= similarity_threshold
            
            # Merge if conditions are met
            if (consecutive_pages or same_page) and (similar_structure or headers_match):
                # Skip headers in the second table if it matches the first
                next_data = next_table['data']
                if headers_match and next_data:
                    next_data = next_data[1:]  # Skip header row in next table
                
                # Combine data
                current_table['data'].extend(next_data)
                
                # Update metadata
                current_table['metadata']['merged_with'] = next_table['table_number']
                current_table['metadata']['spans_pages'] = list(set(
                    current_table['metadata'].get('spans_pages', [current_table['page']]) + 
                    [next_table['page']]
                ))
                
                # Update shape
                rows = len(current_table['data'])
                cols = len(current_table['data'][0]) if rows > 0 else 0
                current_table['metadata']['shape'] = (rows, cols)
            else:
                # Tables don't match, keep the current one and start a new one
                merged_tables.append(current_table)
                current_table = next_table
        
        # Add the last table
        merged_tables.append(current_table)
        
        return merged_tables
    
    def extract_tables_from_parser_output(self, pdf_path: str, parser_output: Dict[str, Any], 
                                         use_direct_camelot: bool = False) -> List[Dict[str, Any]]:
        """
        Extract tables using information from the PDF Parser output.
        
        Args:
            pdf_path: Path to the PDF file
            parser_output: Output from PDFParser.extract_all()
            use_direct_camelot: If True, use Camelot directly for table detection without relying on pre-identified locations
            
        Returns:
            List of dictionaries containing extracted table data
        """
        # Extract table locations from parser output
        table_locations = []
        if not use_direct_camelot and "visual_elements" in parser_output and "tables" in parser_output["visual_elements"]:
            table_locations = parser_output["visual_elements"]["tables"]
        
        # Extract tables using the identified locations or direct Camelot
        tables = self.extract_tables(pdf_path, table_locations, use_direct_camelot=use_direct_camelot)
        
        # Augment tables with additional context from parser output if available
        if tables:
            self._augment_tables_with_context(tables, parser_output)
        
        # Filter out non-tabular content with enhanced criteria
        tables = self.filter_non_tabular_content(tables)
        
        # Apply additional validation to further filter non-tabular content
        tables = self.validate_tables(tables)
        
        return tables
    
    def filter_non_tabular_content(self, tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter out non-tabular content from extracted tables with improved criteria.
        
        Args:
            tables: List of dictionaries containing table data
            
        Returns:
            List of dictionaries containing only true tabular data
        """
        if not self.filter_non_tabular:
            return tables
            
        filtered_tables = []
        
        for table in tables:
            data = table["data"]
            
            # Skip empty tables or tables with empty data
            if not data:
                logger.info(f"Filtered out empty table on page {table.get('page', 'unknown')}")
                continue
                
            # Skip if too small to be a meaningful table
            if len(data) < self.min_table_rows or (data and len(data[0]) < self.min_table_cols):
                logger.info(f"Filtered out small grid on page {table['page']}")
                continue
            
            # Calculate cell content density
            total_cells = sum(len(row) for row in data)
            if total_cells == 0:
                logger.info(f"Filtered out table with no cells on page {table['page']}")
                continue
                
            filled_cells = sum(sum(1 for cell in row if cell and str(cell).strip()) for row in data)
            cell_density = filled_cells / total_cells
            
            # Skip if too many empty cells
            if cell_density < self.min_cell_density:
                logger.info(f"Filtered out sparse grid on page {table['page']}")
                continue
            
            # Check for column/row regularity (tabular data tends to have consistent structure)
            col_counts = [len(row) for row in data]
            if max(col_counts) - min(col_counts) > self.max_column_variation:
                logger.info(f"Filtered out irregular grid on page {table['page']}")
                continue
            
            # Check for text-like content (paragraphs) vs. tabular content
            avg_cell_length = sum(len(str(cell)) for row in data for cell in row if cell) / max(filled_cells, 1)
            
            # Tables typically have shorter cell content
            if avg_cell_length > 100:  # Long text in cells suggests paragraphs, not tables
                logger.info(f"Filtered out text-heavy content on page {table['page']}")
                continue
            
            # Check for balanced column widths (tables tend to have somewhat balanced columns)
            col_width_var = 0
            if data and len(data[0]) > 1:
                # Estimate column widths based on content length
                col_widths = []
                for col_idx in range(len(data[0])):
                    col_content = [row[col_idx] for row in data if col_idx < len(row)]
                    avg_width = sum(len(str(cell)) for cell in col_content) / len(col_content)
                    col_widths.append(avg_width)
                
                # Calculate coefficient of variation for column widths
                mean_width = sum(col_widths) / len(col_widths)
                if mean_width > 0:
                    std_dev = (sum((w - mean_width) ** 2 for w in col_widths) / len(col_widths)) ** 0.5
                    cv = std_dev / mean_width
                    col_width_var = cv
                    
                    # If columns have wildly different widths, it might not be a table
                    if cv > 1.5:  # High variation in column widths
                        logger.info(f"Filtered out content with unbalanced columns on page {table['page']}")
                        continue
                
                # Check for multi-column layout patterns
                if self.is_multi_column_layout(data, col_width_var):
                    logger.info(f"Filtered out multi-column document layout on page {table['page']}")
                    continue
            
            # Check for numeric content percentage (tables often contain numeric data)
            numeric_cells = sum(1 for row in data for cell in row 
                              if cell and str(cell).strip() and re.search(r'\d', str(cell)))
            numeric_percentage = numeric_cells / max(filled_cells, 1)
            
            # If confidence is borderline and numeric content is low, be more strict
            if table.get("metadata", {}).get("accuracy", 100) < 85 and numeric_percentage < 0.2:
                confidence_boost = numeric_percentage * 10  # 0-2 boost based on numeric content
                if table.get("metadata", {}).get("accuracy", 100) + confidence_boost < 85:
                    logger.info(f"Filtered out low-confidence, text-heavy content on page {table['page']}")
                    continue
            
            # Check if content is paragraph-like
            if self.is_paragraph_like(data):
                logger.info(f"Filtered out paragraph-like content on page {table['page']}")
                continue
                
            # Check for consistent table structure
            if not self.has_consistent_table_structure(data):
                logger.info(f"Filtered out content with inconsistent column structure on page {table['page']}")
                continue
            
            # Keep tables that pass all checks
            filtered_tables.append(table)
        
        logger.info(f"Filtered out {len(tables) - len(filtered_tables)} non-tabular elements")
        return filtered_tables
    
    def validate_tables(self, tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate extracted tables using additional heuristics to filter out non-tabular content.
        
        Args:
            tables: List of dictionaries containing table data
            
        Returns:
            List of validated tables
        """
        validated_tables = []
        
        # Get validation parameters from config
        min_numeric_percentage = self.config.get("min_numeric_percentage", 0.25)  # Reduced from 0.3
        max_long_text_percentage = self.config.get("max_long_text_percentage", 0.2)
        min_rows = self.config.get("min_table_rows", 4)
        min_cols = self.config.get("min_table_cols", 2)
        min_cell_density = self.config.get("min_cell_density", 0.7)  # Increased from 0.6
        max_column_variation = self.config.get("max_column_variation", 0.5)  # Reduced from 0.8
        max_layout_variance = self.config.get("max_layout_variance", 0.05)  # New parameter
        
        for table in tables:
            data = table["data"]
            
            # Skip empty tables
            if not data or len(data) == 0:
                logger.info(f"Validation: Skipping empty table on page {table.get('page', 'unknown')}")
                continue
            
            # Calculate table statistics
            rows = len(data)
            cols = max(len(row) for row in data) if data else 0
            
            # Skip very small tables (likely not real tables)
            if rows < min_rows or cols < min_cols:
                logger.info(f"Validation: Skipping small table ({rows}x{cols}) on page {table.get('page', 'unknown')}")
                continue
            
            # Check for consistent delimiters in cells (common in tables)
            delimiter_pattern = re.compile(r'[|,;:]')
            delimiter_cells = sum(1 for row in data for cell in row 
                                if cell and delimiter_pattern.search(str(cell)))
            
            # Check for numeric content (tables often contain numbers)
            numeric_pattern = re.compile(r'\d')
            numeric_cells = sum(1 for row in data for cell in row 
                              if cell and numeric_pattern.search(str(cell)))
            
            # Calculate percentages
            total_cells = rows * cols
            delimiter_percentage = delimiter_cells / total_cells if total_cells > 0 else 0
            numeric_percentage = numeric_cells / total_cells if total_cells > 0 else 0
            
            # Check for long text content (paragraphs)
            long_text_cells = sum(1 for row in data for cell in row 
                                if cell and len(str(cell)) > 100)
            long_text_percentage = long_text_cells / total_cells if total_cells > 0 else 0
            
            # Add additional checks for text/table patterns
            # Check for sentence structure in cells (suggests non-tabular content)
            sentence_pattern = re.compile(r'[A-Z][^.!?]+(\.|\!|\?)')
            sentence_cells = sum(1 for row in data for cell in row 
                               if cell and sentence_pattern.search(str(cell)))
            sentence_percentage = sentence_cells / total_cells if total_cells > 0 else 0
            
            # Check for column width consistency (tables tend to have consistent column widths)
            col_widths = []
            for c in range(cols):
                col_cells = [row[c] for row in data if c < len(row)]
                avg_width = sum(len(str(cell)) for cell in col_cells if cell) / max(sum(1 for cell in col_cells if cell), 1)
                col_widths.append(avg_width)
            
            col_width_var = 0
            if col_widths and sum(col_widths) > 0:
                mean_width = sum(col_widths) / len(col_widths)
                col_width_var = sum((w - mean_width) ** 2 for w in col_widths) / len(col_widths) / (mean_width ** 2)
            
            # NEW: Check for section headings pattern (common in reports)
            section_heading_pattern = re.compile(r'^[A-Z\s]{3,}$')  # All caps text
            section_headings = sum(1 for row in data for cell in row 
                                if cell and section_heading_pattern.match(str(cell)))
            section_heading_ratio = section_headings / total_cells if total_cells > 0 else 0
            
            # NEW: Check for document layout patterns vs. true tables
            if cols >= 3 and rows > 10:
                # Multi-column text typically has very consistent column widths
                if col_width_var < max_layout_variance and sentence_percentage > 0.05:
                    # This is likely a multi-column layout, not a table
                    table["metadata"]["is_likely_document_layout"] = True
                    logger.info(f"Detected multi-column document layout on page {table.get('page', 'unknown')}")
                    continue
            
            # Annual reports often have section headings in all caps
            if section_heading_ratio > 0.05:
                logger.info(f"Detected document with section headings on page {table.get('page', 'unknown')}")
                continue
            
            # Validate based on combined criteria
            is_valid = (
                (numeric_percentage >= min_numeric_percentage or delimiter_percentage > 0.15) and
                long_text_percentage <= max_long_text_percentage and
                sentence_percentage < 0.4 and
                col_width_var < 1.5
            )
            
            # Add validation metrics to table metadata for debugging
            table["metadata"]["validation_metrics"] = {
                "rows": rows,
                "cols": cols,
                "numeric_percentage": numeric_percentage,
                "delimiter_percentage": delimiter_percentage,
                "long_text_percentage": long_text_percentage,
                "sentence_percentage": sentence_percentage,
                "column_width_variance": col_width_var,
                "section_heading_ratio": section_heading_ratio
            }
            
            if is_valid:
                validated_tables.append(table)
            else:
                logger.info(f"Validation: Filtered out non-tabular content on page {table.get('page', 'unknown')} - "
                           f"numeric: {numeric_percentage:.2f}, long_text: {long_text_percentage:.2f}, "
                           f"sentences: {sentence_percentage:.2f}, col_var: {col_width_var:.2f}")
        
        logger.info(f"Validation: Kept {len(validated_tables)} out of {len(tables)} tables")
        return validated_tables
    
    def is_paragraph_like(self, table_data):
        """
        Detects if table content appears to be paragraphs rather than tabular data.
        
        Args:
            table_data: The table data to analyze
            
        Returns:
            Boolean indicating if the content appears to be paragraph-like
        """
        if not table_data:
            return False
            
        # Calculate average words per cell
        total_words = 0
        total_cells = 0
        
        for row in table_data:
            for cell in row:
                if cell and str(cell).strip():
                    words = len(str(cell).split())
                    total_words += words
                    total_cells += 1
        
        avg_words_per_cell = total_words / max(total_cells, 1)
        
        # Check for sentence endings in cells
        sentence_endings = sum(1 for row in table_data for cell in row 
                            if cell and re.search(r'[.!?]\s*$', str(cell)))
        
        # Paragraph-like content typically has more words per cell and sentence endings
        return avg_words_per_cell > 15 or sentence_endings > len(table_data) * 0.5
    
    def is_multi_column_layout(self, data, col_width_var):
        """Detect multi-column document layouts that aren't true tables"""
        if not data or len(data) < 10 or len(data[0]) < 3:
            return False
        
        # Multi-column layouts often have very consistent column widths
        if col_width_var < 0.05:
            # Check for content patterns typical of document layouts
            
            # Text paragraphs across multiple columns
            paragraph_breaks = 0
            for row in data:
                for cell in row:
                    if cell and str(cell).strip() and str(cell).endswith('.'):
                        paragraph_breaks += 1
            
            # Document layouts often have paragraph breaks
            if paragraph_breaks > len(data) * 0.1:
                return True
                
            # Check for section headers patterns (ALL CAPS or Title Case)
            section_headers = 0
            for row in data:
                for cell in row:
                    if cell and (str(cell).isupper() or 
                                str(cell).istitle()):
                        section_headers += 1
            
            if section_headers > len(data) * 0.05:
                return True
        
        return False
    
    def has_consistent_table_structure(self, data):
        """
        Check if the data has a consistent structure typical of tables.
        
        Args:
            data: The table data to analyze
            
        Returns:
            Boolean indicating if the data has a consistent table structure
        """
        if not data or not data[0]:
            return False
            
        # Check column consistency - tables typically have consistent data types per column
        consistent_columns = True
        for col_idx in range(len(data[0])):
            # Get column values
            col_values = [row[col_idx] for row in data if col_idx < len(row)]
            
            # Count different types in column
            numeric_count = sum(1 for val in col_values if val and re.match(r'^[\d.,]+$', str(val)))
            text_count = sum(1 for val in col_values if val and not re.match(r'^[\d.,]+$', str(val)))
            
            # Calculate type consistency
            total = numeric_count + text_count
            if total > 0:
                major_type_ratio = max(numeric_count, text_count) / total
                if major_type_ratio < 0.7:  # Less than 70% consistency
                    consistent_columns = False
                    break
        
        return consistent_columns
    
    def _augment_tables_with_context(self, tables: List[Dict[str, Any]], parser_output: Dict[str, Any]) -> None:
        """
        Augment table data with additional context from the parser output.
        
        Args:
            tables: List of dictionaries containing table data
            parser_output: Output from PDFParser.extract_all()
        """
        # Get page content from parser output
        page_content = {}
        if "text" in parser_output and "pages" in parser_output["text"]:
            for page in parser_output["text"]["pages"]:
                page_num = page.get("page_number")
                if page_num:
                    page_content[page_num] = page.get("content", "")
        
        # Augment each table with surrounding context
        for table in tables:
            page_num = table.get("page")
            if page_num in page_content:
                # Try to find table caption or title in the page content
                content = page_content[page_num]
                
                # Look for table references above the table
                table_refs = [
                    r"Table\s+\d+[\.:]\s*([^\n]+)",
                    r"TABLE\s+\d+[\.:]\s*([^\n]+)"
                ]
                
                for pattern in table_refs:
                    matches = list(re.finditer(pattern, content))
                    if matches:
                        # Use the last match before the table as the caption
                        # This is a simplification - in reality, we would need to
                        # consider the position of the table on the page
                        caption = matches[-1].group(1).strip()
                        table["metadata"]["caption"] = caption
                        break
                
                # Add a snippet of text before and after the table for context
                # This is a simplified approach - in a real implementation, we would
                # use the table's position on the page to extract relevant context
                if "rect" in table:
                    # If we have the table's position, we could extract text around it
                    # This is just a placeholder for a more sophisticated implementation
                    pass
                else:
                    # Without position, just add some general page context
                    # Limit to a reasonable size to avoid bloating the output
                    max_context_len = 200
                    if len(content) > max_context_len:
                        context = content[:max_context_len] + "..."
                    else:
                        context = content
                    
                    table["metadata"]["page_context"] = context
