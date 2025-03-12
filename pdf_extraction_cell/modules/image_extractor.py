"""
Image Extraction Module for extracting images from PDF documents.
Modified to better handle complex images and charts.
"""
import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import io
import json

# Import image libraries with fallback options
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import pdf2image
except ImportError:
    pdf2image = None

logger = logging.getLogger("pdf_extraction.image_extractor")

class ImageExtractor:
    """Class for extracting images from PDF documents using various methods."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the image extractor with configuration.
        
        Args:
            config: Configuration dictionary for the extractor
        """
        self.config = config
        self.extraction_method = config.get("image_extraction_method", "auto")
        # Reduce minimum image size to catch more charts/graphs
        self.min_image_size = config.get("min_image_size", 50)  # Reduced from 100
        self.output_format = config.get("image_output_format", "png")
        # Increase DPI for better resolution
        self.output_dpi = config.get("image_output_dpi", 600)  # Increased from 300
        self.extract_vector_graphics = config.get("extract_vector_graphics", True)
        self.save_image_metadata = config.get("save_image_metadata", True)
        # Add new option to extract entire pages as images
        self.extract_pages_as_images = config.get("extract_pages_as_images", True)
        # Add fallback extraction option
        self.use_fallback_extraction = config.get("use_fallback_extraction", True)
        
        self._validate_dependencies()
        logger.info(f"Initialized Image Extractor with method: {self.extraction_method}")
    
    def _validate_dependencies(self) -> None:
        """Validate that required dependencies are available based on the chosen method."""
        if self.extraction_method == "pymupdf" and fitz is None:
            logger.warning("PyMuPDF (fitz) is not installed. Falling back to another method.")
            if pdf2image is not None and Image is not None:
                self.extraction_method = "pdf2image"
            else:
                logger.error("No image extraction libraries available.")
                self.extraction_method = "none"
                
        elif self.extraction_method == "pdf2image" and (pdf2image is None or Image is None):
            logger.warning("pdf2image or Pillow is not installed. Falling back to another method.")
            if fitz is not None:
                self.extraction_method = "pymupdf"
            else:
                logger.error("No image extraction libraries available.")
                self.extraction_method = "none"
                
        elif self.extraction_method == "auto":
            # Choose the best available method (prefer PyMuPDF over pdf2image)
            if fitz is not None:
                self.extraction_method = "pymupdf"
            elif pdf2image is not None and Image is not None:
                self.extraction_method = "pdf2image"
            else:
                logger.error("No image extraction libraries available.")
                self.extraction_method = "none"
    
    def extract_images(self, pdf_path: str, 
                      image_locations: Optional[List[Dict[str, Any]]] = None, 
                      page_range: Optional[Union[str, List[int]]] = None,
                      output_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Extract images from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            image_locations: Optional list of potential image locations identified by the PDF Parser
            page_range: Optional page range to extract images from
            output_dir: Directory to save extracted images (defaults to same directory as PDF)
            
        Returns:
            List of dictionaries containing extracted image metadata
        """
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file does not exist: {pdf_path}")
            return []
            
        logger.info(f"Extracting images from {pdf_path} using {self.extraction_method}")
        
        if self.extraction_method == "none":
            logger.warning("No image extraction method available. Returning empty result.")
            return []
        
        # Set output directory if not provided
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(pdf_path), "extracted_images")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine pages to process
        if page_range is None and image_locations:
            # Extract images only from pages where potential images were identified
            pages = sorted(set(loc["page"] for loc in image_locations if "page" in loc))
            if not pages:
                page_range = "all"
            else:
                page_range = ",".join(str(p) for p in pages)
        
        images_data = []
        
        # Try primary extraction method
        if self.extraction_method == "pymupdf":
            images_data = self._extract_images_pymupdf(pdf_path, image_locations, page_range, output_dir)
        elif self.extraction_method == "pdf2image":
            images_data = self._extract_images_pdf2image(pdf_path, image_locations, page_range, output_dir)
        
        # Always extract pages as images using pdf2image as a fallback for charts/tables
        if self.extract_pages_as_images and self.use_fallback_extraction and pdf2image is not None:
            logger.info("Also extracting pages as high-resolution images")
            page_images = self._extract_pages_as_images(pdf_path, page_range, output_dir)
            images_data.extend(page_images)
        
        # Save image metadata to JSON if requested
        if self.save_image_metadata and images_data:
            metadata_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(pdf_path))[0]}_images.json")
            self._save_metadata_to_json(images_data, metadata_path)
        
        return images_data
    
    def _extract_pages_as_images(self, pdf_path: str, 
                               page_range: Optional[Union[str, List[int]]],
                               output_dir: str) -> List[Dict[str, Any]]:
        """Extract each page as a high-resolution image."""
        images_data = []
        
        try:
            # Determine which pages to process
            if isinstance(page_range, str) and page_range == "all":
                pages_to_process = None  # pdf2image will process all pages
            elif isinstance(page_range, str):
                # Convert string like "1-3,5,7-9" to a list of integers
                pages_to_process = []
                for part in page_range.split(','):
                    if '-' in part:
                        start, end = map(int, part.split('-'))
                        pages_to_process.extend(range(start, end + 1))  # 1-indexed
                    else:
                        pages_to_process.append(int(part))  # 1-indexed
            elif isinstance(page_range, list):
                pages_to_process = page_range  # Assuming 1-indexed
            else:
                pages_to_process = None  # Process all pages
            
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            
            # Use a higher DPI for better quality, but cap it to avoid exceeding API limits
            high_dpi = max(600, self.output_dpi)
            
            # Convert PDF pages to images using pdf2image
            pil_images = pdf2image.convert_from_path(
                pdf_path,
                dpi=high_dpi,
                first_page=min(pages_to_process) if pages_to_process else None,
                last_page=max(pages_to_process) if pages_to_process else None,
                output_folder=None,  # Don't save yet
                fmt="png",  # Always use PNG for highest quality
                thread_count=4  # Use multiple threads for faster conversion
            )
            
            # Maximum allowed dimension for Claude API (8000 pixels)
            MAX_IMAGE_DIMENSION = 8000
            
            # Save each page as a high-quality image
            for i, img in enumerate(pil_images):
                page_num = (min(pages_to_process) + i) if pages_to_process else (i + 1)
                output_filename = f"{base_name}_page{page_num}_high_res.png"
                output_path = os.path.join(output_dir, output_filename)
                
                # Check if image dimensions exceed the maximum allowed
                width, height = img.size
                if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
                    logger.info(f"Image dimensions ({width}x{height}) exceed maximum allowed ({MAX_IMAGE_DIMENSION}). Resizing...")
                    
                    # Calculate new dimensions while maintaining aspect ratio
                    if width >= height:
                        new_width = MAX_IMAGE_DIMENSION
                        new_height = int(height * (MAX_IMAGE_DIMENSION / width))
                    else:
                        new_height = MAX_IMAGE_DIMENSION
                        new_width = int(width * (MAX_IMAGE_DIMENSION / height))
                    
                    # Resize the image
                    img = img.resize((new_width, new_height), Image.LANCZOS)
                    logger.info(f"Resized image to {new_width}x{new_height}")
                
                img.save(output_path, quality=100, optimize=True)
                
                # Record metadata (use img.size to get the potentially resized dimensions)
                image_data = {
                    "page": page_num,
                    "filename": output_filename,
                    "path": output_path,
                    "width": img.width,
                    "height": img.height,
                    "format": "png",
                    "dpi": high_dpi,
                    "extraction_method": "high_res_page_image"
                }
                
                images_data.append(image_data)
                logger.info(f"Extracted high-res page {page_num} as {output_filename} ({img.width}x{img.height})")
            
        except Exception as e:
            logger.error(f"Error extracting pages as images: {str(e)}")
        
        return images_data
    
    def _extract_images_pymupdf(self, pdf_path: str, 
                              image_locations: Optional[List[Dict[str, Any]]], 
                              page_range: Optional[Union[str, List[int]]],
                              output_dir: str) -> List[Dict[str, Any]]:
        """Extract images using PyMuPDF (fitz)."""
        images_data = []
        
        try:
            doc = fitz.open(pdf_path)
            
            # Determine which pages to process
            if isinstance(page_range, str) and page_range == "all":
                pages_to_process = range(len(doc))
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
                pages_to_process = range(len(doc))
            
            image_count = 0
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            
            # Process each page
            for page_idx in pages_to_process:
                if page_idx >= len(doc):
                    logger.warning(f"Page {page_idx+1} out of range, skipping")
                    continue
                
                page = doc[page_idx]
                
                # Extract images using get_images
                image_list = page.get_images(full=True)
                
                # Also extract embedded images with get_text("dict") - might catch more images
                text_dict = page.get_text("dict")
                if "images" in text_dict:
                    for img_info in text_dict["images"]:
                        # Check if this image is already in our list
                        xref = img_info.get("xref")
                        if xref and not any(img[0] == xref for img in image_list):
                            # Add this image to our list
                            image_list.append((xref, None, None, None, None, None, None, None))
                
                for img_idx, img_info in enumerate(image_list):
                    xref = img_info[0]  # XRef number of the image
                    
                    # Extract the image
                    base_image = doc.extract_image(xref)
                    if not base_image:
                        continue
                    
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    width = base_image.get("width", 0)
                    height = base_image.get("height", 0)
                    
                    # Skip small images if configured to do so
                    if width < self.min_image_size or height < self.min_image_size:
                        logger.info(f"Skipping small image ({width}x{height}) on page {page_idx+1}")
                        continue
                    
                    # Generate output filename
                    image_count += 1
                    output_ext = self.output_format.lower()
                    output_filename = f"{base_name}_page{page_idx+1}_img{image_count}.{output_ext}"
                    output_path = os.path.join(output_dir, output_filename)
                    
                    # Convert and save the image
                    try:
                        # If output format is different from original, convert using PIL
                        if output_ext != image_ext.lower() and Image is not None:
                            pil_image = Image.open(io.BytesIO(image_bytes))
                            pil_image.save(output_path, quality=100, optimize=True)
                        else:
                            # Save directly if format matches
                            with open(output_path, "wb") as f:
                                f.write(image_bytes)
                        
                        # Record metadata
                        image_data = {
                            "page": page_idx + 1,  # 1-indexed for consistency
                            "image_number": image_count,
                            "filename": output_filename,
                            "path": output_path,
                            "width": width,
                            "height": height,
                            "format": image_ext,
                            "xref": xref,
                            "extraction_method": "pymupdf"
                        }
                        
                        # Add additional metadata if available
                        if "colorspace" in base_image:
                            image_data["colorspace"] = base_image["colorspace"]
                        if "cs-name" in base_image:
                            image_data["colorspace_name"] = base_image["cs-name"]
                        
                        images_data.append(image_data)
                        logger.info(f"Extracted image {output_filename} ({width}x{height})")
                        
                    except Exception as e:
                        logger.error(f"Error saving image {output_filename}: {str(e)}")
                
                # Check if this page has images
                has_images = len(image_list) > 0
                
                # Extract vector graphics if enabled
                has_vector = False
                if self.extract_vector_graphics:
                    try:
                        # Save page as SVG
                        svg_filename = f"{base_name}_page{page_idx+1}_vector.svg"
                        svg_path = os.path.join(output_dir, svg_filename)
                        
                        svg = page.get_svg_image()
                        with open(svg_path, "w", encoding="utf-8") as f:
                            f.write(svg)
                        
                        # Record metadata
                        vector_data = {
                            "page": page_idx + 1,
                            "filename": svg_filename,
                            "path": svg_path,
                            "width": page.rect.width,
                            "height": page.rect.height,
                            "format": "svg",
                            "extraction_method": "pymupdf_vector"
                        }
                        
                        images_data.append(vector_data)
                        logger.info(f"Extracted vector graphics to {svg_filename}")
                        has_vector = True
                        
                    except Exception as e:
                        logger.error(f"Error extracting vector graphics from page {page_idx+1}: {str(e)}")
                
                # Also always render the page as a PNG image (enhanced for charts/tables)
                try:
                    # Set the zoom factor to achieve a higher DPI for better quality
                    # PyMuPDF uses 72 dpi as the base resolution
                    zoom = self.output_dpi / 72
                    
                    # Create a transformation matrix with the zoom factor
                    mat = fitz.Matrix(zoom, zoom)
                    
                    # Render the page to a pixmap with better quality settings
                    pix = page.get_pixmap(matrix=mat, alpha=False, colorspace="rgb")
                    
                    # Maximum allowed dimension for Claude API (8000 pixels)
                    MAX_IMAGE_DIMENSION = 8000
                    
                    # Check if pixmap dimensions exceed the maximum allowed
                    if pix.width > MAX_IMAGE_DIMENSION or pix.height > MAX_IMAGE_DIMENSION:
                        logger.info(f"Pixmap dimensions ({pix.width}x{pix.height}) exceed maximum allowed ({MAX_IMAGE_DIMENSION}). Adjusting zoom...")
                        
                        # Calculate scale factor needed to fit within maximum dimensions
                        scale_factor = min(MAX_IMAGE_DIMENSION / pix.width, MAX_IMAGE_DIMENSION / pix.height)
                        
                        # Apply scale factor to original zoom
                        adjusted_zoom = zoom * scale_factor
                        
                        # Create a new transformation matrix with the adjusted zoom
                        adjusted_mat = fitz.Matrix(adjusted_zoom, adjusted_zoom)
                        
                        # Re-render the page with the adjusted zoom
                        pix = page.get_pixmap(matrix=adjusted_mat, alpha=False, colorspace="rgb")
                        logger.info(f"Adjusted pixmap dimensions to {pix.width}x{pix.height}")
                        
                        # Double-check that dimensions are now within limits
                        if pix.width > MAX_IMAGE_DIMENSION or pix.height > MAX_IMAGE_DIMENSION:
                            logger.warning(f"Pixmap still exceeds maximum dimensions ({pix.width}x{pix.height}). Forcing resize...")
                            # Force resize to maximum allowed dimensions
                            if pix.width >= pix.height:
                                new_width = MAX_IMAGE_DIMENSION
                                new_height = int(pix.height * (MAX_IMAGE_DIMENSION / pix.width))
                            else:
                                new_height = MAX_IMAGE_DIMENSION
                                new_width = int(pix.width * (MAX_IMAGE_DIMENSION / pix.height))
                            
                            # Create a new pixmap with the correct dimensions
                            new_pix = fitz.Pixmap(pix.colorspace, (0, 0, new_width, new_height), pix.alpha)
                            new_pix.set_origin(pix.x, pix.y)
                            pix.shrink(new_width / pix.width)
                            pix = new_pix
                            logger.info(f"Forced resize to {pix.width}x{pix.height}")
                    
                    # Save the pixmap as a PNG
                    png_filename = f"{base_name}_page{page_idx+1}_full.png"
                    png_path = os.path.join(output_dir, png_filename)
                    pix.save(png_path)
                    
                    # Record metadata
                    png_data = {
                        "page": page_idx + 1,
                        "filename": png_filename,
                        "path": png_path,
                        "width": pix.width,
                        "height": pix.height,
                        "format": "png",
                        "dpi": self.output_dpi,
                        "extraction_method": "pymupdf_full_page"
                    }
                    
                    images_data.append(png_data)
                    logger.info(f"Rendered full page {page_idx+1} as PNG: {png_filename} ({pix.width}x{pix.height})")
                    
                except Exception as e:
                    logger.error(f"Error rendering full page {page_idx+1} as PNG: {str(e)}")
            
            doc.close()
            logger.info(f"Successfully extracted {len(images_data)} images from {pdf_path}")
            
        except Exception as e:
            logger.error(f"Error extracting images with PyMuPDF: {str(e)}")
            # Try fallback method
            if self.use_fallback_extraction and pdf2image is not None and Image is not None:
                logger.info("Falling back to pdf2image for image extraction")
                return self._extract_images_pdf2image(pdf_path, image_locations, page_range, output_dir)
            else:
                return []
        
        return images_data
    
    def _extract_images_pdf2image(self, pdf_path: str, 
                                image_locations: Optional[List[Dict[str, Any]]], 
                                page_range: Optional[Union[str, List[int]]],
                                output_dir: str) -> List[Dict[str, Any]]:
        """Extract images using pdf2image and Pillow."""
        images_data = []
        
        try:
            # Determine which pages to process
            if isinstance(page_range, str) and page_range == "all":
                pages_to_process = None  # pdf2image will process all pages
            elif isinstance(page_range, str):
                # Convert string like "1-3,5,7-9" to a list of integers
                pages_to_process = []
                for part in page_range.split(','):
                    if '-' in part:
                        start, end = map(int, part.split('-'))
                        pages_to_process.extend(range(start, end + 1))  # 1-indexed
                    else:
                        pages_to_process.append(int(part))  # 1-indexed
            elif isinstance(page_range, list):
                pages_to_process = page_range  # Assuming 1-indexed
            else:
                pages_to_process = None  # Process all pages
            
            # Convert page range to pdf2image format
            if pages_to_process:
                first_page = min(pages_to_process)
                last_page = max(pages_to_process)
                
                # Check if pages are consecutive
                if list(range(first_page, last_page + 1)) == pages_to_process:
                    # Consecutive pages - use first_page and last_page
                    pdf2image_pages = (first_page, last_page)
                else:
                    # Non-consecutive pages - convert each page separately
                    pdf2image_pages = None  # Will handle below
            else:
                pdf2image_pages = None  # All pages
            
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            
            # Use a higher DPI for better quality
            dpi = self.output_dpi
            
            # Convert PDF pages to images
            if pdf2image_pages:
                # Convert consecutive pages
                pil_images = pdf2image.convert_from_path(
                    pdf_path,
                    dpi=dpi,
                    first_page=pdf2image_pages[0],
                    last_page=pdf2image_pages[1],
                    output_folder=None,  # Don't save yet
                    fmt="png",  # Always use PNG for highest quality
                    thread_count=4,  # Use multiple threads for faster conversion
                    use_cropbox=True  # Use cropbox for more accurate extraction
                )
                
                # Save each page as an image
                for i, img in enumerate(pil_images):
                    page_num = pdf2image_pages[0] + i
                    output_filename = f"{base_name}_page{page_num}.png"
                    output_path = os.path.join(output_dir, output_filename)
                    
                    # Maximum allowed dimension for Claude API (8000 pixels)
                    MAX_IMAGE_DIMENSION = 8000
                    
                    # Check if image dimensions exceed the maximum allowed
                    width, height = img.size
                    if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
                        logger.info(f"Image dimensions ({width}x{height}) exceed maximum allowed ({MAX_IMAGE_DIMENSION}). Resizing...")
                        
                        # Calculate new dimensions while maintaining aspect ratio
                        if width >= height:
                            new_width = MAX_IMAGE_DIMENSION
                            new_height = int(height * (MAX_IMAGE_DIMENSION / width))
                        else:
                            new_height = MAX_IMAGE_DIMENSION
                            new_width = int(width * (MAX_IMAGE_DIMENSION / height))
                        
                        # Resize the image
                        img = img.resize((new_width, new_height), Image.LANCZOS)
                        logger.info(f"Resized image to {new_width}x{new_height}")
                    
                    img.save(output_path, quality=100, optimize=True)
                    
                    # Record metadata
                    image_data = {
                        "page": page_num,
                        "image_number": i + 1,
                        "filename": output_filename,
                        "path": output_path,
                        "width": img.width,
                        "height": img.height,
                        "format": "png",
                        "dpi": dpi,
                        "extraction_method": "pdf2image_page"
                    }
                    
                    images_data.append(image_data)
                    logger.info(f"Extracted page {page_num} as {output_filename} ({img.width}x{img.height})")
                
            elif pages_to_process:
                # Convert each page separately
                for page_num in pages_to_process:
                    try:
                        pil_images = pdf2image.convert_from_path(
                            pdf_path,
                            dpi=dpi,
                            first_page=page_num,
                            last_page=page_num,
                            output_folder=None,  # Don't save yet
                            fmt="png",  # Always use PNG for highest quality
                            use_cropbox=True  # Use cropbox for more accurate extraction
                        )
                        
                        if not pil_images:
                            logger.warning(f"No image generated for page {page_num}")
                            continue
                        
                        img = pil_images[0]
                        output_filename = f"{base_name}_page{page_num}.png"
                        output_path = os.path.join(output_dir, output_filename)
                        
                        # Maximum allowed dimension for Claude API (8000 pixels)
                        MAX_IMAGE_DIMENSION = 8000
                        
                        # Check if image dimensions exceed the maximum allowed
                        width, height = img.size
                        if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
                            logger.info(f"Image dimensions ({width}x{height}) exceed maximum allowed ({MAX_IMAGE_DIMENSION}). Resizing...")
                            
                            # Calculate new dimensions while maintaining aspect ratio
                            if width >= height:
                                new_width = MAX_IMAGE_DIMENSION
                                new_height = int(height * (MAX_IMAGE_DIMENSION / width))
                            else:
                                new_height = MAX_IMAGE_DIMENSION
                                new_width = int(width * (MAX_IMAGE_DIMENSION / height))
                            
                            # Resize the image
                            img = img.resize((new_width, new_height), Image.LANCZOS)
                            logger.info(f"Resized image to {new_width}x{new_height}")
                        
                        img.save(output_path, quality=100, optimize=True)
                        
                        # Record metadata
                        image_data = {
                            "page": page_num,
                            "image_number": 1,
                            "filename": output_filename,
                            "path": output_path,
                            "width": img.width,
                            "height": img.height,
                            "format": "png",
                            "dpi": dpi,
                            "extraction_method": "pdf2image_page"
                        }
                        
                        images_data.append(image_data)
                        logger.info(f"Extracted page {page_num} as {output_filename} ({img.width}x{img.height})")
                        
                    except Exception as e:
                        logger.error(f"Error extracting page {page_num}: {str(e)}")
            
            else:
                # Convert all pages
                pil_images = pdf2image.convert_from_path(
                    pdf_path,
                    dpi=dpi,
                    output_folder=None,  # Don't save yet
                    fmt="png",  # Always use PNG for highest quality
                    thread_count=4,  # Use multiple threads for faster conversion
                    use_cropbox=True  # Use cropbox for more accurate extraction
                )
                
                # Save each page as an image
                for i, img in enumerate(pil_images):
                    page_num = i + 1  # 1-indexed
                    output_filename = f"{base_name}_page{page_num}.png"
                    output_path = os.path.join(output_dir, output_filename)
                    
                    # Maximum allowed dimension for Claude API (8000 pixels)
                    MAX_IMAGE_DIMENSION = 8000
                    
                    # Check if image dimensions exceed the maximum allowed
                    width, height = img.size
                    if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
                        logger.info(f"Image dimensions ({width}x{height}) exceed maximum allowed ({MAX_IMAGE_DIMENSION}). Resizing...")
                        
                        # Calculate new dimensions while maintaining aspect ratio
                        if width >= height:
                            new_width = MAX_IMAGE_DIMENSION
                            new_height = int(height * (MAX_IMAGE_DIMENSION / width))
                        else:
                            new_height = MAX_IMAGE_DIMENSION
                            new_width = int(width * (MAX_IMAGE_DIMENSION / height))
                        
                        # Resize the image
                        img = img.resize((new_width, new_height), Image.LANCZOS)
                        logger.info(f"Resized image to {new_width}x{new_height}")
                    
                    img.save(output_path, quality=100, optimize=True)
                    
                    # Record metadata
                    image_data = {
                        "page": page_num,
                        "image_number": 1,
                        "filename": output_filename,
                        "path": output_path,
                        "width": img.width,
                        "height": img.height,
                        "format": "png",
                        "dpi": dpi,
                        "extraction_method": "pdf2image_page"
                    }
                    
                    images_data.append(image_data)
                    logger.info(f"Extracted page {page_num} as {output_filename} ({img.width}x{img.height})")
            
            logger.info(f"Successfully extracted {len(images_data)} images from {pdf_path}")
            
        except Exception as e:
            logger.error(f"Error extracting images with pdf2image: {str(e)}")
            return []
        
        return images_data
    
    def _save_metadata_to_json(self, images_data: List[Dict[str, Any]], output_path: str) -> None:
        """
        Save image metadata to a JSON file.
        
        Args:
            images_data: List of dictionaries containing image metadata
            output_path: Path to save the JSON file
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(images_data, f, indent=2)
            logger.info(f"Saved image metadata to {output_path}")
        except Exception as e:
            logger.error(f"Error saving image metadata to JSON: {str(e)}")
    
    def extract_images_from_parser_output(self, pdf_path: str, parser_output: Dict[str, Any],
                                         output_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Extract images using information from the PDF Parser output.
        
        Args:
            pdf_path: Path to the PDF file
            parser_output: Output from PDFParser.extract_all()
            output_dir: Directory to save extracted images (defaults to same directory as PDF)
            
        Returns:
            List of dictionaries containing extracted image data
        """
        # Extract image locations from parser output
        image_locations = []
        if "visual_elements" in parser_output and "images" in parser_output["visual_elements"]:
            image_locations = parser_output["visual_elements"]["images"]
        
        # Also check for tables, figures, and charts that should be treated as images
        if "visual_elements" in parser_output:
            if "tables" in parser_output["visual_elements"]:
                for table in parser_output["visual_elements"]["tables"]:
                    if "page" in table:
                        image_locations.append(table)
            
            if "figures" in parser_output["visual_elements"]:
                for figure in parser_output["visual_elements"]["figures"]:
                    if "page" in figure:
                        image_locations.append(figure)
            
            if "charts" in parser_output["visual_elements"]:
                for chart in parser_output["visual_elements"]["charts"]:
                    if "page" in chart:
                        image_locations.append(chart)
        
        # Extract images using the identified locations
        images = self.extract_images(pdf_path, image_locations, output_dir=output_dir)
        
        return images
