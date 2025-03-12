"""
PDF Parser Module for extracting text and metadata from PDF documents.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import re

# Import PDF libraries with fallback options
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    from pypdf import PdfReader
except ImportError:
    try:
        from PyPDF2 import PdfReader  # Fallback to older version
    except ImportError:
        PdfReader = None

logger = logging.getLogger("pdf_extraction.pdf_parser")

class PDFParser:
    """
    Class for parsing PDF documents and extracting text and metadata.
    Supports multiple PDF libraries with fallback mechanisms.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the PDF parser with configuration.
        
        Args:
            config: Configuration dictionary with extraction settings
        """
        self.config = config
        self.extraction_method = config.get("text_extraction_method", "auto")
        self._validate_dependencies()
        logger.info(f"Initialized PDF Parser with method: {self.extraction_method}")

    def _validate_dependencies(self) -> None:
        """Validate that required dependencies are available based on the chosen method."""
        if self.extraction_method == "pymupdf" and fitz is None:
            logger.warning("PyMuPDF (fitz) is not installed. Falling back to pypdf.")
            self.extraction_method = "pypdf"
        
        if self.extraction_method == "pypdf" and PdfReader is None:
            logger.warning("PyPDF/PyPDF2 is not installed. Falling back to PyMuPDF.")
            self.extraction_method = "pymupdf"
            
        if self.extraction_method == "auto":
            if PdfReader is not None:
                self.extraction_method = "pypdf"
            elif fitz is not None:
                self.extraction_method = "pymupdf"
            else:
                raise ImportError("No PDF library available. Please install pypdf or PyMuPDF.")
    
    def extract_all(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract all content from a PDF file including text, metadata, and markers for images and tables.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing all extracted content and markers
        """
        logger.info(f"Extracting all content from {pdf_path}")
        
        result = {}
        
        # Extract text with page structure
        text_data = self.extract_text(pdf_path)
        result["text"] = text_data
        
        # Extract metadata
        metadata = self.extract_metadata(pdf_path)
        result["metadata"] = metadata
        
        # Identify potential tables and images
        if self.extraction_method == "pymupdf":
            visual_elements = self._identify_visual_elements_pymupdf(pdf_path)
        else:
            # For pypdf, we'll use a simpler approach
            visual_elements = self._identify_visual_elements_heuristic(text_data)
        
        result["visual_elements"] = visual_elements
        
        logger.info(f"Successfully extracted all content from {pdf_path}")
        return result
    
    def extract_text(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract text from a PDF file with page markers.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing extracted text organized by pages
        """
        logger.info(f"Extracting text from {pdf_path} using {self.extraction_method}")
        
        if self.extraction_method == "pymupdf":
            return self._extract_text_pymupdf(pdf_path)
        elif self.extraction_method == "pypdf":
            return self._extract_text_pypdf(pdf_path)
        else:
            raise ValueError(f"Unsupported extraction method: {self.extraction_method}")
    
    def _extract_text_pymupdf(self, pdf_path: str) -> Dict[str, Any]:
        """Extract text using PyMuPDF (fitz)."""
        result = {"pages": []}
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num, page in enumerate(doc, 1):
                text = page.get_text()
                
                # Process text (clean up, fix encoding issues, etc.)
                text = self._clean_text(text)
                
                # Add to result
                result["pages"].append({
                    "page_number": page_num,
                    "content": text,
                    "layout": {
                        "width": page.rect.width,
                        "height": page.rect.height
                    }
                })
            
            # Add doc-level metadata
            result["page_count"] = len(doc)
            
            doc.close()
            logger.info(f"Successfully extracted text from {pdf_path} using PyMuPDF")
            
        except Exception as e:
            logger.error(f"Error extracting text with PyMuPDF: {str(e)}")
            raise
        
        return result
    
    def _extract_text_pypdf(self, pdf_path: str) -> Dict[str, Any]:
        """Extract text using pypdf/PyPDF2."""
        result = {"pages": []}
        
        try:
            with open(pdf_path, "rb") as file:
                reader = PdfReader(file)
                
                for page_num, page in enumerate(reader.pages, 1):
                    text = page.extract_text()
                    
                    # Process text (clean up, fix encoding issues, etc.)
                    text = self._clean_text(text)
                    
                    # Add to result
                    result["pages"].append({
                        "page_number": page_num,
                        "content": text
                    })
                
                # Add doc-level metadata
                result["page_count"] = len(reader.pages)
            
            logger.info(f"Successfully extracted text from {pdf_path} using PyPDF")
            
        except Exception as e:
            logger.error(f"Error extracting text with PyPDF: {str(e)}")
            raise
        
        return result
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text to fix common issues."""
        if not text:
            return ""
            
        # Replace multiple newlines with a single one
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Fix common encoding issues
        text = text.replace('\u2028', '\n')  # Line separator
        text = text.replace('\u2029', '\n\n')  # Paragraph separator
        
        # Remove excessive whitespace
        text = re.sub(r' {2,}', ' ', text)
        text = text.strip()
        
        return text
    
    def extract_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract metadata from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing document metadata
        """
        logger.info(f"Extracting metadata from {pdf_path} using {self.extraction_method}")
        
        if self.extraction_method == "pymupdf":
            return self._extract_metadata_pymupdf(pdf_path)
        elif self.extraction_method == "pypdf":
            return self._extract_metadata_pypdf(pdf_path)
        else:
            raise ValueError(f"Unsupported extraction method: {self.extraction_method}")
    
    def _extract_metadata_pymupdf(self, pdf_path: str) -> Dict[str, Any]:
        """Extract metadata using PyMuPDF (fitz)."""
        metadata = {}
        
        try:
            doc = fitz.open(pdf_path)
            
            # Basic metadata
            metadata = {
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "subject": doc.metadata.get("subject", ""),
                "creator": doc.metadata.get("creator", ""),
                "producer": doc.metadata.get("producer", ""),
                "keywords": doc.metadata.get("keywords", ""),
                "creation_date": doc.metadata.get("creationDate", ""),
                "modification_date": doc.metadata.get("modDate", ""),
                "page_count": len(doc)
            }
            
            # Additional document info
            metadata["has_toc"] = True if doc.get_toc() else False
            
            # Clean up date strings - they come in a special format
            for date_field in ["creation_date", "modification_date"]:
                if metadata[date_field] and metadata[date_field].startswith("D:"):
                    # Convert PDF date format to ISO-like
                    date_str = metadata[date_field][2:]  # Remove D: prefix
                    try:
                        # Basic format: YYYYMMDDHHmmSS
                        year = date_str[0:4]
                        month = date_str[4:6]
                        day = date_str[6:8]
                        metadata[date_field] = f"{year}-{month}-{day}"
                    except:
                        # If parsing fails, keep original
                        pass
            
            doc.close()
            logger.info(f"Successfully extracted metadata from {pdf_path} using PyMuPDF")
            
        except Exception as e:
            logger.error(f"Error extracting metadata with PyMuPDF: {str(e)}")
            raise
        
        return metadata
    
    def _extract_metadata_pypdf(self, pdf_path: str) -> Dict[str, Any]:
        """Extract metadata using pypdf/PyPDF2."""
        metadata = {}
        
        try:
            with open(pdf_path, "rb") as file:
                reader = PdfReader(file)
                
                # Get document info dictionary
                info = reader.metadata
                if info:
                    metadata = {
                        "title": info.get("/Title", ""),
                        "author": info.get("/Author", ""),
                        "subject": info.get("/Subject", ""),
                        "creator": info.get("/Creator", ""),
                        "producer": info.get("/Producer", ""),
                        "keywords": info.get("/Keywords", ""),
                        "creation_date": info.get("/CreationDate", ""),
                        "modification_date": info.get("/ModDate", ""),
                        "page_count": len(reader.pages)
                    }
                    
                    # Clean up values - remove potential null bytes and convert from PDF strings
                    for key, value in metadata.items():
                        if isinstance(value, str):
                            metadata[key] = value.replace("\x00", "").strip()
                
                # Set page count even if no metadata was found
                if not metadata:
                    metadata = {"page_count": len(reader.pages)}
            
            logger.info(f"Successfully extracted metadata from {pdf_path} using PyPDF")
            
        except Exception as e:
            logger.error(f"Error extracting metadata with PyPDF: {str(e)}")
            raise
        
        return metadata
    
    def _identify_visual_elements_pymupdf(self, pdf_path: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Identify potential tables and images using PyMuPDF.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary with lists of potential tables and images
        """
        result = {
            "tables": [],
            "images": []
        }
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num, page in enumerate(doc, 1):
                # Find images
                image_list = page.get_images(full=True)
                
                for img_idx, img_info in enumerate(image_list, 1):
                    xref = img_info[0]  # XRef number of the image
                    
                    # Get the image properties
                    base_image = doc.extract_image(xref)
                    image_ext = base_image["ext"]  # Extension (e.g., 'png', 'jpeg')
                    
                    # Get position for the image - need to find the reference in page blocks
                    image_rect = None
                    for block in page.get_text("dict")["blocks"]:
                        if block["type"] == 1:  # Image blocks
                            # Found an image block
                            image_rect = block["bbox"]  # [x0, y0, x1, y1]
                            break
                    
                    result["images"].append({
                        "page": page_num,
                        "image_number": img_idx,
                        "xref": xref,
                        "extension": image_ext,
                        "rect": image_rect,
                        "extraction_method": "pymupdf"
                    })
                
                # Find potential tables using heuristics
                # Check for tabular patterns in text blocks
                blocks = page.get_text("dict")["blocks"]
                
                # Simple table detection: look for blocks with aligned text
                table_candidates = []
                
                # Analyze blocks for potential table patterns
                # This is a simplified approach - real table detection is more complex
                for block_idx, block in enumerate(blocks):
                    if block["type"] == 0:  # Text block
                        lines = block.get("lines", [])
                        
                        # Check if block has multiple lines with similar structure
                        if len(lines) >= 3:  # At least 3 lines for a table
                            # Get spans per line
                            spans_per_line = [sum(1 for span in line.get("spans", [])) for line in lines]
                            
                            # If all lines have similar number of spans, it might be a table
                            if len(set(spans_per_line)) <= 2 and sum(spans_per_line) >= len(lines) * 2:
                                table_candidates.append({
                                    "page": page_num,
                                    "block_number": block_idx,
                                    "rect": block["bbox"],
                                    "line_count": len(lines),
                                    "confidence": 0.7,  # Simple confidence score
                                    "extraction_method": "pymupdf"
                                })
                
                # Add potential tables to the result
                result["tables"].extend(table_candidates)
            
            doc.close()
            logger.info(f"Identified {len(result['images'])} images and {len(result['tables'])} potential tables")
            
        except Exception as e:
            logger.error(f"Error identifying visual elements: {str(e)}")
            raise
        
        return result
    
    def _identify_visual_elements_heuristic(self, text_data: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Identify potential tables and images using text-based heuristics.
        
        Args:
            text_data: Dictionary containing extracted text
            
        Returns:
            Dictionary with lists of potential tables and images
        """
        result = {
            "tables": [],
            "images": []
        }
        
        # Simple heuristics for table detection based on text patterns
        table_patterns = [
            r'Table\s+\d+',  # "Table 1", "Table 2", etc.
            r'[\|\+][-+]+[\|\+]',  # ASCII tables with pipe and dash characters
            r'(\s*[\w\d.,%$]+\s*\|){2,}',  # Content with multiple pipe separators
            r'([\w\d.,%$]+\t){2,}',  # Tab-separated content
            r'(\d+[,.]\d+\s+){3,}'  # Multiple aligned numbers (financial tables)
        ]
        
        # Simple heuristics for image detection based on text patterns
        image_patterns = [
            r'Figure\s+\d+',  # "Figure 1", "Figure 2", etc.
            r'Chart\s+\d+',  # "Chart 1", "Chart 2", etc.
            r'Graph\s+\d+',  # "Graph 1", "Graph 2", etc.
            r'Diagram\s+\d+'  # "Diagram 1", "Diagram 2", etc.
        ]
        
        for page in text_data.get("pages", []):
            page_num = page.get("page_number", 0)
            content = page.get("content", "")
            
            # Find potential tables
            for pattern in table_patterns:
                for match in re.finditer(pattern, content, re.MULTILINE):
                    # Get context around the match
                    start, end = match.span()
                    line_start = content.rfind('\n', 0, start) + 1
                    line_end = content.find('\n', end)
                    if line_end == -1:
                        line_end = len(content)
                    
                    # Extract lines around the match for context
                    context_start = max(0, line_start - 100)
                    context_end = min(len(content), line_end + 100)
                    context = content[context_start:context_end]
                    
                    result["tables"].append({
                        "page": page_num,
                        "match": match.group(0),
                        "context": context,
                        "position": (start, end),
                        "confidence": 0.6,  # Lower confidence for heuristic detection
                        "extraction_method": "text_heuristic"
                    })
            
            # Find potential images
            for pattern in image_patterns:
                for match in re.finditer(pattern, content, re.MULTILINE):
                    # Get context around the match
                    start, end = match.span()
                    line_start = content.rfind('\n', 0, start) + 1
                    line_end = content.find('\n', end)
                    if line_end == -1:
                        line_end = len(content)
                    
                    # Extract lines around the match for context
                    context_start = max(0, line_start - 50)
                    context_end = min(len(content), line_end + 50)
                    context = content[context_start:context_end]
                    
                    result["images"].append({
                        "page": page_num,
                        "match": match.group(0),
                        "context": context,
                        "position": (start, end),
                        "confidence": 0.5,  # Lower confidence for text-based image detection
                        "extraction_method": "text_heuristic"
                    })
        
        logger.info(f"Identified {len(result['images'])} potential images and {len(result['tables'])} potential tables using heuristics")
        return result
