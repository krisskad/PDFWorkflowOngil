"""
Table Post-Processing Module using Gemini Flash Lite 2 via OpenRouter for improving table extraction results.
This script processes the output from TableExtractor and enhances it with
contextual information from surrounding text.
"""

import json
import logging
import os
import re
from typing import Dict, Any, List, Optional, Union, Tuple
from openai import OpenAI
import time
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
project_root = script_dir.parent
env_path = project_root / '.env'
if env_path.exists():
    load_dotenv(env_path)

# Configure logging
logger = logging.getLogger("pdf_extraction.table_post_processor")

class TablePostProcessor:
    """
    Class for post-processing table extraction results and enhancing them with contextual information.
    Improves table formatting and structure based on context from the text content.
    """
    
    def __init__(self, config: Dict[str, Any], company_name: Optional[str] = None):
        """
        Initialize the table post-processor with configuration.
        
        Args:
            config: Configuration dictionary with processing options and API settings
            company_name: Optional company name for context-aware processing
        """
        self.config = config
        self.company_name = company_name
        
        # Extract configuration options
        self.fix_headers = config.get("fix_headers", True)
        self.fix_alignment = config.get("fix_alignment", True)
        self.detect_merged_cells = config.get("detect_merged_cells", True)
        self.extract_table_titles = config.get("extract_table_titles", True)
        self.extract_footnotes = config.get("extract_footnotes", True)
        self.dedup_tables = config.get("dedup_tables", True)
        self.clean_empty_rows_cols = config.get("clean_empty_rows_cols", True)
        self.context_window_size = config.get("context_window_size", 1)  # Pages before/after to include as context
        
        # Enhanced analysis settings
        self.summarize_tables = config.get("summarize_tables", True)
        self.extract_keywords = config.get("extract_keywords", True)
        self.extract_insights = config.get("extract_insights", True)
        self.detect_table_type = config.get("detect_table_type", True)
        
        # Optional custom prompts
        self.custom_prompts = config.get("custom_prompts", {})
        
        # API configuration (if using Gemini to assist with complex transformations)
        self.use_gemini_api = config.get("use_gemini_api", False)
        if self.use_gemini_api:
            # Try to get API key from config, environment variable, or .env file
            self.api_key = (
                config.get("openrouter_api_key") or 
                os.environ.get("OPENROUTER_API_KEY")
            )
            
            if not self.api_key:
                logger.warning("OpenRouter API key not provided. Complex transformations will use rule-based approaches only.")
                self.use_gemini_api = False
            else:
                # Initialize OpenAI client with OpenRouter configuration
                self.model = config.get("gemini_model") or os.environ.get("GEMINI_MODEL", "google/gemini-2.0-flash-lite-001")
                self.max_tokens = int(config.get("max_tokens") or os.environ.get("MAX_TOKENS", "4096"))
                self.temperature = float(config.get("temperature") or os.environ.get("TEMPERATURE", "0.0"))
                self.rate_limit_delay = config.get("rate_limit_delay", 1.0)  # Seconds between API calls
                
                # Site information for OpenRouter
                self.site_url = config.get("site_url") or os.environ.get("SITE_URL", "https://example.com")
                self.site_name = config.get("site_name") or os.environ.get("SITE_NAME", "PDF Extraction Tool")
                
                # Initialize OpenAI client with OpenRouter configuration
                self.client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=self.api_key,
                )
        
        logger.info(f"Initialized Table Post-Processor with use_gemini_api={self.use_gemini_api}")
        if self.company_name:
            logger.info(f"Processing tables for company: {self.company_name}")
    
    def process_tables(self, 
                      tables: List[Dict[str, Any]], 
                      enhanced_text: Optional[Dict[str, Any]] = None,
                      output_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Process and enhance table extraction results using surrounding text context.
        
        Args:
            tables: List of dictionaries containing extracted table data from TableExtractor
            enhanced_text: Optional enhanced text output from TextPostProcessor for context
            output_path: Optional path to save the enhanced tables
            
        Returns:
            List of enhanced table dictionaries
        """
        logger.info(f"Starting post-processing of {len(tables)} tables")
        
        # Create a deep copy to avoid modifying the original data
        enhanced_tables = tables.copy()
        
        # Skip if no tables to process
        if not enhanced_tables:
            logger.info("No tables to process")
            return enhanced_tables
        
        # Clean and fix basic table structure issues for all tables
        if self.clean_empty_rows_cols:
            enhanced_tables = self._clean_empty_rows_and_columns(enhanced_tables)
            logger.info("Cleaned empty rows and columns")
            
            # Save intermediate results after cleaning
            if output_path:
                self._save_enhanced_tables(enhanced_tables, output_path)
                logger.info(f"Saved intermediate tables after cleaning to {output_path}")
        
        # Extract text content by page from enhanced_text if available
        page_text_content = {}
        if enhanced_text and "original_content" in enhanced_text:
            parser_output = enhanced_text.get("original_content", {})
            if "text" in parser_output and "pages" in parser_output["text"]:
                for page in parser_output["text"]["pages"]:
                    page_num = page.get("page_number")
                    if page_num:
                        page_text_content[page_num] = page.get("content", "")
                
                logger.info(f"Extracted text content for {len(page_text_content)} pages")
        
        # Process each table
        for i, table in enumerate(enhanced_tables):
            logger.info(f"Processing table {i+1} on page {table.get('page', 'unknown')}")
            
            # Get surrounding page context
            context = self._get_page_context(table, page_text_content)
            
            # Skip tables with no data
            if not table.get("data"):
                logger.warning(f"Skipping table {i+1} on page {table.get('page', 'unknown')} - no data")
                continue
            
            # Fix table headers if needed
            if self.fix_headers:
                table = self._fix_table_headers(table, context)
            
            # Fix alignment issues
            if self.fix_alignment:
                table = self._fix_table_alignment(table, context)
            
            # Detect and handle merged cells
            if self.detect_merged_cells:
                table = self._handle_merged_cells(table, context)
            
            # Extract table title from context
            if self.extract_table_titles and context:
                table = self._extract_table_title(table, context)
            
            # Extract table footnotes from context
            if self.extract_footnotes and context:
                table = self._extract_table_footnotes(table, context)
            
            # If using Gemini API, apply complex transformations
            if self.use_gemini_api and context:
                table = self._apply_gemini_enhancements(table, context)
                
                # Save intermediate results after Gemini enhancements
                if output_path:
                    # Update the table in the enhanced_tables list
                    enhanced_tables[i] = table
                    self._save_enhanced_tables(enhanced_tables, output_path)
                    logger.info(f"Saved intermediate tables after Gemini enhancements for table {i+1} to {output_path}")
            
            # Apply enhanced analysis if Gemini API is available
            if self.use_gemini_api:
                # Initialize AI enhancements if not present
                if "ai_enhancements" not in table:
                    table["ai_enhancements"] = {}
                
                # Check which enhancements are enabled
                enhancements_needed = []
                if self.detect_table_type:
                    enhancements_needed.append("table_type")
                if self.summarize_tables:
                    enhancements_needed.append("summary")
                if self.extract_keywords:
                    enhancements_needed.append("keywords")
                if self.extract_insights:
                    enhancements_needed.append("insights")
                
                # If any enhancements are needed, make a single API call
                if enhancements_needed:
                    enhancements = self._get_combined_table_enhancements(table, context, enhancements_needed)
                    
                    # Add each enhancement to the table
                    for enhancement_type, enhancement_data in enhancements.items():
                        table["ai_enhancements"][enhancement_type] = enhancement_data
                    
                    logger.info(f"Applied AI enhancements ({', '.join(enhancements_needed)}) for table {i+1} with a single API call")
                    
                    # Save intermediate results after AI enhancements
                    if output_path:
                        # Update the table in the enhanced_tables list
                        enhanced_tables[i] = table
                        self._save_enhanced_tables(enhanced_tables, output_path)
                        logger.info(f"Saved intermediate tables after AI enhancements for table {i+1} to {output_path}")
            
            # Apply any custom analysis defined in the config
            if self.use_gemini_api:
                if self.custom_prompts:
                    logger.info(f"Found {len(self.custom_prompts)} custom prompts to apply: {', '.join(self.custom_prompts.keys())}")
                    
                    # Initialize AI enhancements if not present
                    if "ai_enhancements" not in table:
                        table["ai_enhancements"] = {}
                    
                    # Apply each custom prompt
                    for analysis_name, prompt in self.custom_prompts.items():
                        logger.info(f"Applying custom analysis '{analysis_name}' for table {i+1}")
                        custom_analysis = self._run_custom_analysis(table, context, analysis_name, prompt)
                        table["ai_enhancements"][analysis_name] = custom_analysis
                        logger.info(f"Applied custom analysis '{analysis_name}' for table {i+1}")
                        
                        # Save intermediate results after custom analysis
                        if output_path:
                            # Update the table in the enhanced_tables list
                            enhanced_tables[i] = table
                            self._save_enhanced_tables(enhanced_tables, output_path)
                            logger.info(f"Saved intermediate tables after custom analysis '{analysis_name}' for table {i+1} to {output_path}")
                else:
                    logger.info("No custom prompts found in configuration")
            
            # Update the table in the enhanced_tables list
            enhanced_tables[i] = table
        
        # Remove duplicate tables if needed
        if self.dedup_tables:
            enhanced_tables = self._deduplicate_tables(enhanced_tables)
            logger.info(f"Deduplicated tables: {len(enhanced_tables)} remaining")
            
            # Save intermediate results after deduplication
            if output_path:
                self._save_enhanced_tables(enhanced_tables, output_path)
                logger.info(f"Saved intermediate tables after deduplication to {output_path}")
        
        # Final save of enhanced tables if output path provided
        if output_path:
            self._save_enhanced_tables(enhanced_tables, output_path)
            logger.info(f"Saved final enhanced tables to {output_path}")
        
        return enhanced_tables
    
    def _get_page_context(self, table: Dict[str, Any], page_text_content: Dict[int, str]) -> Dict[str, str]:
        """
        Get contextual information from surrounding pages.
        
        Args:
            table: Table dictionary
            page_text_content: Dictionary mapping page numbers to text content
            
        Returns:
            Dictionary with current, previous, and next page context
        """
        context = {}
        
        # Get current page number
        page_num = table.get("page")
        if not page_num:
            logger.warning(f"Table missing page number")
            return context
        
        # Get current page context
        if page_num in page_text_content:
            context["current_page"] = page_text_content[page_num]
        
        # Get previous pages based on context window size
        for i in range(1, self.context_window_size + 1):
            prev_page = page_num - i
            if prev_page in page_text_content:
                context[f"previous_page_{i}"] = page_text_content[prev_page]
        
        # Get next pages based on context window size
        for i in range(1, self.context_window_size + 1):
            next_page = page_num + i
            if next_page in page_text_content:
                context[f"next_page_{i}"] = page_text_content[next_page]
        
        return context
    
    def _clean_empty_rows_and_columns(self, tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Clean empty rows and columns from tables.
        
        Args:
            tables: List of table dictionaries
            
        Returns:
            List of tables with empty rows and columns removed
        """
        cleaned_tables = []
        
        for table in tables:
            data = table.get("data", [])
            if not data:
                cleaned_tables.append(table)
                continue
            
            # Remove empty rows
            non_empty_rows = []
            for row in data:
                if any(cell and str(cell).strip() for cell in row):
                    non_empty_rows.append(row)
            
            if not non_empty_rows:
                # Skip tables with no data after cleaning
                logger.warning(f"Table on page {table.get('page', 'unknown')} has no non-empty rows after cleaning")
                cleaned_tables.append(table)
                continue
            
            # Find empty columns (all cells in column are empty)
            num_cols = max(len(row) for row in non_empty_rows)
            empty_cols = []
            
            for col_idx in range(num_cols):
                col_is_empty = True
                for row in non_empty_rows:
                    if col_idx < len(row) and row[col_idx] and str(row[col_idx]).strip():
                        col_is_empty = False
                        break
                
                if col_is_empty:
                    empty_cols.append(col_idx)
            
            # Remove empty columns
            if empty_cols:
                cleaned_rows = []
                for row in non_empty_rows:
                    cleaned_row = [cell for i, cell in enumerate(row) if i not in empty_cols]
                    cleaned_rows.append(cleaned_row)
                
                # Update table data
                table["data"] = cleaned_rows
                
                # Update metadata if needed
                if "metadata" in table and "shape" in table["metadata"]:
                    new_shape = (len(cleaned_rows), len(cleaned_rows[0]) if cleaned_rows else 0)
                    table["metadata"]["shape"] = new_shape
                    table["metadata"]["cleaned_empty_rows_cols"] = True
            else:
                # Just update with non-empty rows
                table["data"] = non_empty_rows
                
                # Update metadata if needed
                if "metadata" in table and "shape" in table["metadata"]:
                    new_shape = (len(non_empty_rows), len(non_empty_rows[0]) if non_empty_rows else 0)
                    table["metadata"]["shape"] = new_shape
                    table["metadata"]["cleaned_empty_rows"] = True
            
            cleaned_tables.append(table)
            
        return cleaned_tables
    
    def _fix_table_headers(self, table: Dict[str, Any], context: Dict[str, str]) -> Dict[str, Any]:
        """
        Fix or improve table headers based on content and context.
        
        Args:
            table: Table dictionary
            context: Text context from surrounding pages
            
        Returns:
            Table with improved headers
        """
        data = table.get("data", [])
        if not data or len(data) < 2:
            return table
        
        # Check if headers are already defined in metadata
        metadata = table.get("metadata", {})
        has_headers = "headers" in metadata and metadata["headers"]
        
        # If no headers defined, try to detect them from first row
        if not has_headers:
            first_row = data[0]
            
            # Check if first row looks like a header
            is_header_row = True
            
            # Headers typically have all cells filled
            if not all(cell and str(cell).strip() for cell in first_row):
                is_header_row = False
            
            # Headers typically have shorter text than data rows
            avg_first_row_len = sum(len(str(cell)) for cell in first_row) / len(first_row)
            avg_data_row_len = 0
            data_row_count = 0
            
            for row in data[1:]:
                if any(cell and str(cell).strip() for cell in row):
                    avg_data_row_len += sum(len(str(cell)) for cell in row) / len(row)
                    data_row_count += 1
            
            if data_row_count > 0:
                avg_data_row_len /= data_row_count
                if avg_first_row_len > avg_data_row_len * 1.5:
                    # Headers are usually shorter than data rows
                    is_header_row = False
            
            # If first row looks like a header, set it as header
            if is_header_row:
                if "metadata" not in table:
                    table["metadata"] = {}
                
                table["metadata"]["headers"] = first_row
                table["metadata"]["has_detected_header"] = True
                # Remove header row from data to avoid duplication
                table["data"] = data[1:]
                logger.info(f"Detected and set header row for table on page {table.get('page', 'unknown')}")
        
        # Clean up existing headers
        if "headers" in table.get("metadata", {}):
            headers = table["metadata"]["headers"]
            
            # Clean up headers (remove newlines, normalize whitespace)
            clean_headers = [
                re.sub(r'\s+', ' ', str(h).replace('\n', ' ')).strip() if h else "" 
                for h in headers
            ]
            
            # Update headers
            table["metadata"]["headers"] = clean_headers
            table["metadata"]["headers_cleaned"] = True
        
        return table
    
    def _fix_table_alignment(self, table: Dict[str, Any], context: Dict[str, str]) -> Dict[str, Any]:
        """
        Fix table alignment issues (columns misaligned, etc.).
        
        Args:
            table: Table dictionary
            context: Text context from surrounding pages
            
        Returns:
            Table with improved alignment
        """
        data = table.get("data", [])
        if not data:
            return table
        
        # Normalize row lengths
        max_cols = max(len(row) for row in data)
        
        normalized_data = []
        for row in data:
            if len(row) < max_cols:
                # Pad row with empty strings
                normalized_row = row + [""] * (max_cols - len(row))
                normalized_data.append(normalized_row)
            elif len(row) > max_cols:
                # Trim row to max_cols
                normalized_row = row[:max_cols]
                normalized_data.append(normalized_row)
            else:
                normalized_data.append(row)
        
        # Update table data
        table["data"] = normalized_data
        
        # Update metadata
        if "metadata" in table:
            table["metadata"]["shape"] = (len(normalized_data), max_cols)
            table["metadata"]["alignment_fixed"] = True
        
        return table
    
    def _handle_merged_cells(self, table: Dict[str, Any], context: Dict[str, str]) -> Dict[str, Any]:
        """
        Detect and handle merged cells in tables.
        
        Args:
            table: Table dictionary
            context: Text context from surrounding pages
            
        Returns:
            Table with properly handled merged cells
        """
        data = table.get("data", [])
        if not data:
            return table
        
        # Check for vertical merges (same value repeated in consecutive rows in the same column)
        merged_cells = []
        
        for col_idx in range(len(data[0])):
            consecutive_same = 1
            last_value = None
            
            for row_idx, row in enumerate(data):
                if col_idx >= len(row):
                    continue
                
                current_value = row[col_idx]
                current_value_str = str(current_value).strip() if current_value else ""
                
                if current_value_str == last_value and current_value_str:
                    consecutive_same += 1
                    # Mark this cell as part of a vertical merge
                    if consecutive_same > 1:
                        merged_cells.append((row_idx, col_idx, "vertical"))
                else:
                    # Reset counter when value changes
                    consecutive_same = 1
                    last_value = current_value_str
        
        # Update table metadata with merged cell information
        if merged_cells:
            if "metadata" not in table:
                table["metadata"] = {}
            
            table["metadata"]["merged_cells"] = merged_cells
            table["metadata"]["has_merged_cells"] = True
            logger.info(f"Detected {len(merged_cells)} merged cells in table on page {table.get('page', 'unknown')}")
        
        return table
    
    def _extract_table_title(self, table: Dict[str, Any], context: Dict[str, str]) -> Dict[str, Any]:
        """
        Extract table title from surrounding text context.
        
        Args:
            table: Table dictionary
            context: Text context from surrounding pages
            
        Returns:
            Table with extracted title
        """
        if not context:
            return table
        
        # Patterns to identify table titles
        title_patterns = [
            r'Table\s+\d+\s*[\.:]?\s*([^\n]+)',
            r'TABLE\s+\d+\s*[\.:]?\s*([^\n]+)',
            r'Table\s+\d+\-\d+\s*[\.:]?\s*([^\n]+)',
            r'TABLE\s+\d+\-\d+\s*[\.:]?\s*([^\n]+)'
        ]
        
        # Look for table title in current page context
        current_page_text = context.get("current_page", "")
        
        # Initialize variables to track the best match
        table_title = None
        best_score = 0
        
        # Try each pattern to find table titles
        for pattern in title_patterns:
            matches = re.finditer(pattern, current_page_text)
            for match in matches:
                title = match.group(1).strip()
                if title:
                    # Calculate a relevance score based on quality of match
                    score = len(title)
                    if score > best_score:
                        best_score = score
                        table_title = title
        
        # If no match in current page, try previous page
        if not table_title and "previous_page_1" in context:
            prev_page_text = context.get("previous_page_1", "")
            for pattern in title_patterns:
                matches = re.finditer(pattern, prev_page_text)
                for match in matches:
                    title = match.group(1).strip()
                    if title:
                        # Calculate a relevance score
                        score = len(title) * 0.8  # Lower priority for previous page
                        if score > best_score:
                            best_score = score
                            table_title = title
        
        # Add title to metadata if found
        if table_title:
            if "metadata" not in table:
                table["metadata"] = {}
            
            table["metadata"]["title"] = table_title
            logger.info(f"Extracted title for table on page {table.get('page', 'unknown')}: {table_title}")
        
        return table
    
    def _extract_table_footnotes(self, table: Dict[str, Any], context: Dict[str, str]) -> Dict[str, Any]:
        """
        Extract table footnotes from surrounding text context.
        
        Args:
            table: Table dictionary
            context: Text context from surrounding pages
            
        Returns:
            Table with extracted footnotes
        """
        if not context:
            return table
        
        # Patterns to identify table footnotes
        footnote_patterns = [
            r'Note[s]?\s*:\s*([^\n]+)',
            r'NOTE[S]?\s*:\s*([^\n]+)',
            r'Source[s]?\s*:\s*([^\n]+)',
            r'SOURCE[S]?\s*:\s*([^\n]+)',
            r'^\*\s+([^\n]+)',
            r'^\d+\.\s+([^\n]+)'
        ]
        
        current_page_text = context.get("current_page", "")
        
        # Look for footnotes (usually after the table)
        footnotes = []
        
        for pattern in footnote_patterns:
            matches = re.finditer(pattern, current_page_text, re.MULTILINE)
            for match in matches:
                footnote = match.group(1).strip()
                if footnote:
                    footnotes.append(footnote)
        
        # If no footnotes found in current page, try next page
        if not footnotes and "next_page_1" in context:
            next_page_text = context.get("next_page_1", "")
            for pattern in footnote_patterns:
                matches = re.finditer(pattern, next_page_text, re.MULTILINE)
                for match in matches:
                    footnote = match.group(1).strip()
                    if footnote:
                        footnotes.append(footnote)
        
        # Add footnotes to metadata if found
        if footnotes:
            if "metadata" not in table:
                table["metadata"] = {}
            
            table["metadata"]["footnotes"] = footnotes
            logger.info(f"Extracted {len(footnotes)} footnotes for table on page {table.get('page', 'unknown')}")
        
        return table
    
    def _call_gemini_api(self, prompt: str, system: str = None) -> Dict[str, Any]:
        """
        Call the Gemini API via OpenRouter with the given prompt.
        
        Args:
            prompt: The user message to send to Gemini
            system: Optional system prompt to guide Gemini's behavior
            
        Returns:
            Gemini API response via OpenRouter
        """
        if not self.use_gemini_api:
            return {"error": "Gemini API not enabled"}
            
        if not system:
            # Add company name to system prompt if available
            company_context = f" specializing in {self.company_name} data" if self.company_name else ""
            system = f"You are a helpful AI assistant{company_context} specializing in data analysis and table interpretation."
        
        try:
            # Call the Gemini API via OpenRouter
            response = self.client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": self.site_url,
                    "X-Title": self.site_name,
                },
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": system
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            # Add a delay to respect rate limits
            time.sleep(self.rate_limit_delay)
            
            # Convert the response object to dict for consistency with the rest of the code
            return {
                "content": [{"text": response.choices[0].message.content}],
                "model": response.model,
                "id": response.id,
                "object": response.object
            }
            
        except Exception as e:
            logger.error(f"OpenRouter API call failed: {str(e)}")
            raise
    
    def _parse_json_from_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and parse JSON from Gemini's response.
        
        Args:
            response: Gemini API response
            
        Returns:
            Parsed JSON data
        """
        try:
            content = response.get("content", [])
            if not content:
                return {}
                
            text = content[0].get("text", "")
            
            # Try to extract JSON using regex
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', text)
            if json_match:
                json_str = json_match.group(1)
                return json.loads(json_str)
            
            # If no JSON code block, try to find JSON directly
            json_match = re.search(r'(\{[\s\S]*\})', text)
            if json_match:
                json_str = json_match.group(1)
                return json.loads(json_str)
            
            # If still no JSON, return the full text as a single field
            return {"text": text}
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from response: {str(e)}")
            return {"error": "Failed to parse JSON", "raw_text": text}
    
    def _get_combined_table_enhancements(self, table: Dict[str, Any], context: Dict[str, str], enhancements_needed: List[str]) -> Dict[str, Any]:
        """
        Make a single API call to get all requested table enhancements.
        
        Args:
            table: Table dictionary
            context: Text context from surrounding pages
            enhancements_needed: List of enhancement types to request
            
        Returns:
            Dictionary with all requested enhancements
        """
        if not self.use_gemini_api or not enhancements_needed:
            return {}
            
        # Get table data and headers
        data = table.get("data", [])
        if not data:
            return {enhancement: {"error": "No data to analyze"} for enhancement in enhancements_needed}
            
        headers = table.get("metadata", {}).get("headers", [])
        title = table.get("metadata", {}).get("title", "")
        
        # Format table for prompt
        table_str = "Table Title: " + (title or "Unknown") + "\n\n"
        
        # Add headers if available
        if headers:
            table_str += "Headers: " + " | ".join(str(h) for h in headers) + "\n\n"
        
        # Add data rows (limit to 15 rows to save tokens)
        table_str += "Data:\n"
        for i, row in enumerate(data[:15]):
            table_str += " | ".join(str(cell) for cell in row) + "\n"
        
        if len(data) > 15:
            table_str += f"... (additional {len(data) - 15} rows not shown)\n"
        
        # Add context if available
        context_str = ""
        if context and "current_page" in context:
            context_str = context["current_page"][:1000]  # Limit context length
        
        # Build the system prompt
        company_context = f" with expertise in {self.company_name} data" if self.company_name else ""
        system_prompt = f"""
        You are an expert in data analysis and table interpretation{company_context}. Your task is to analyze tables
        and provide multiple types of analysis in a single response. Structure your response as a JSON object
        with separate sections for each type of analysis requested.
        """
        
        # Build the user prompt based on requested enhancements
        company_info = f" from {self.company_name}" if self.company_name else ""
        prompt = f"""
        Analyze the following table{company_info} and provide the requested information:
        
        {table_str}
        
        Surrounding context:
        ```
        {context_str}
        ```
        
        Please provide the following analyses in a SINGLE JSON object:
        """
        
        # Add specific requests for each enhancement type
        if "table_type" in enhancements_needed:
            prompt += """
            "table_type": {
                "type": The primary table type (e.g., financial, statistical, comparative, reference, etc.),
                "subtypes": Any more specific categorizations (array),
                "purpose": The main purpose/function of this table in the document,
                "data_category": The category of data presented (e.g., financial metrics, demographic data, etc.),
                "confidence": Your confidence level (0-1)
            },
            """
        
        if "summary" in enhancements_needed:
            prompt += """
            "summary": {
                "brief_summary": A 1-2 sentence summary of what this table shows,
                "detailed_summary": A more comprehensive explanation (3-5 sentences),
                "key_observations": An array of 3-5 key observations from the data,
                "data_range": The range or scope of data presented (e.g., time period, categories),
                "audience": Who would find this table most useful
            },
            """
        
        if "keywords" in enhancements_needed:
            prompt += """
            "keywords": {
                "main_topics": An array of 3-5 main topics covered in the table,
                "keywords": An array of 5-10 important keywords from the table,
                "categories": The main categories or dimensions of data presented,
                "entities": Important named entities mentioned (people, organizations, products, etc.)
            },
            """
        
        if "insights" in enhancements_needed:
            prompt += """
            "insights": {
                "key_insights": An array of 3-5 important insights from the data,
                "trends": Any notable trends or patterns observed,
                "anomalies": Any outliers or unusual data points,
                "comparisons": Notable comparisons or contrasts within the data,
                "data_quality_issues": Any potential issues with the data (missing values, inconsistencies)
            }
            """
        
        prompt += """
        
        Respond with ONLY a JSON object containing the requested analyses. No other text.
        """
        
        try:
            # Make a single API call
            response = self._call_gemini_api(prompt, system_prompt)
            result = self._parse_json_from_response(response)
            
            # Ensure all requested enhancements are present in the result
            for enhancement in enhancements_needed:
                if enhancement not in result:
                    result[enhancement] = {"error": f"Failed to generate {enhancement}"}
            
            return result
        except Exception as e:
            logger.error(f"Error during API call for table enhancements: {str(e)}")
            # Return a minimal result with error information
            return {enhancement: {"error": f"API call failed: {str(e)}"} for enhancement in enhancements_needed}
    
    
    def _apply_gemini_enhancements(self, table: Dict[str, Any], context: Dict[str, str]) -> Dict[str, Any]:
        """
        Use Gemini API to enhance table formatting and structure.
        
        Args:
            table: Table dictionary
            context: Text context from surrounding pages
            
        Returns:
            Enhanced table
        """
        if not self.use_gemini_api or not context:
            return table
        
        # Get table data and metadata
        data = table.get("data", [])
        if not data:
            return table
            
        metadata = table.get("metadata", {})
        
        # Format table data for API prompt
        table_str = "```\n"
        
        # Add headers if available
        headers = metadata.get("headers", [])
        if headers:
            table_str += "|" + "|".join(str(h) for h in headers) + "|\n"
            table_str += "|" + "|".join("---" for _ in headers) + "|\n"
        
        # Add data rows
        for row in data:
            table_str += "|" + "|".join(str(cell) for cell in row) + "|\n"
        
        table_str += "```"
        
        # Build context string (current page, limited to 2000 chars for performance)
        current_page_text = context.get("current_page", "")
        context_str = current_page_text[:2000]
        
        # Build prompt for Gemini
        # Add company name to system prompt if available
        company_context = f" for {self.company_name}" if self.company_name else ""
        
        system_prompt = f"""
        You are an expert in table extraction and formatting{company_context}. Your task is to analyze and improve the
        structure of a potentially misaligned or malformatted table. Focus on:
        1. Identifying and proposing corrections for misaligned columns
        2. Detecting merged cells and header spans
        3. Identifying proper table headers if they're mixed with the data
        4. Suggesting corrections for cells that appear to be split incorrectly
        
        Provide your response in JSON format with the corrected table structure.
        DO NOT make up or add data that doesn't exist in the original table.
        Only reorganize and fix formatting issues based on the provided information.
        """
        
        # Add company name to prompt if available
        company_info = f" from {self.company_name}" if self.company_name else ""
        
        prompt = f"""
        I have a table extracted from a PDF{company_info} that may have formatting issues. Please help improve its structure.
        
        The current table structure is:
        {table_str}
        
        Surrounding text that provides context:
        ```
        {context_str}
        ```
        
        Please analyze this table and suggest improvements to its structure based on content and context.
        Focus ONLY on:
        1. Fixing misaligned columns
        2. Identifying merged cells
        3. Correctly formatting headers
        4. Identifying split cells that should be merged
        
        DO NOT add any new data or modify the actual content - only fix structure issues.
        
        Respond with a JSON object containing:
        1. "corrected_data": A 2D array with the properly structured table data
        2. "headers": An array of the correct headers (if identifiable)
        3. "merged_cells": Array of [row, col, span_type] tuples for merged cells
        4. "notes": Brief explanation of major corrections made
        """
        
        try:
            # Call Gemini API
            message = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Add a delay for rate limiting
            time.sleep(self.rate_limit_delay)
            
            # Extract JSON from response
            content = message.content[0].text
            
            # Try to extract JSON using regex
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', content)
            if json_match:
                json_str = json_match.group(1)
                gemini_result = json.loads(json_str)
            else:
                # If no JSON code block, try to find JSON directly
                json_match = re.search(r'(\{[\s\S]*\})', content)
                if json_match:
                    json_str = json_match.group(1)
                    gemini_result = json.loads(json_str)
                else:
                    logger.warning("Could not extract JSON from Gemini response")
                    return table
            
            # Apply Gemini's suggestions to the table
            if "corrected_data" in gemini_result and gemini_result["corrected_data"]:
                table["data"] = gemini_result["corrected_data"]
                if "metadata" not in table:
                    table["metadata"] = {}
                table["metadata"]["gemini_enhanced"] = True
                
                # Update shape
                rows = len(gemini_result["corrected_data"])
                cols = len(gemini_result["corrected_data"][0]) if rows > 0 else 0
                table["metadata"]["shape"] = (rows, cols)
            
            # Update headers if provided
            if "headers" in gemini_result and gemini_result["headers"]:
                table["metadata"]["headers"] = gemini_result["headers"]
                table["metadata"]["headers_gemini_enhanced"] = True
            
            # Add merged cells if identified
            if "merged_cells" in gemini_result and gemini_result["merged_cells"]:
                table["metadata"]["merged_cells"] = gemini_result["merged_cells"]
                table["metadata"]["has_merged_cells"] = True
            
            # Add notes about changes made
            if "notes" in gemini_result and gemini_result["notes"]:
                table["metadata"]["enhancement_notes"] = gemini_result["notes"]
            
            logger.info(f"Applied Gemini enhancements to table on page {table.get('page', 'unknown')}")
            
        except Exception as e:
            logger.error(f"Error calling Gemini API: {str(e)}")
            # Continue without Gemini enhancements
        
        return table
    
    def _deduplicate_tables(self, tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate tables based on content similarity.
        
        Args:
            tables: List of table dictionaries
            
        Returns:
            Deduplicated tables
        """
        if len(tables) <= 1:
            return tables
        
        unique_tables = []
        
        for i, table in enumerate(tables):
            is_duplicate = False
            
            # Compare with previously processed tables
            for unique_table in unique_tables:
                # Skip comparison if tables are from different pages
                if table.get("page") != unique_table.get("page"):
                    continue
                
                # Compare table data
                similarity = self._calculate_table_similarity(table, unique_table)
                
                # If similarity above threshold, mark as duplicate
                if similarity > 0.9:  # 90% similarity threshold
                    is_duplicate = True
                    logger.info(f"Table {i+1} on page {table.get('page', 'unknown')} is a duplicate")
                    break
            
            if not is_duplicate:
                unique_tables.append(table)
        
        return unique_tables
    
    def format_table_using_context(self, table: Dict[str, Any], context: Dict[str, str]) -> Dict[str, Any]:
        """
        Comprehensive method to format table using textual context.
        This method combines multiple formatting steps in an optimal sequence.
        
        Args:
            table: Table dictionary
            context: Text context from surrounding pages
            
        Returns:
            Table with context-aware formatting
        """
        if not table.get("data") or not context:
            return table
            
        # 1. First extract table title and footnotes to understand context
        table = self._extract_table_title(table, context)
        table = self._extract_table_footnotes(table, context)
        
        # 2. Try to infer proper column structure from context
        table = self._infer_column_structure(table, context)
        
        # 3. Clean table (remove empty rows/columns)
        data = table.get("data", [])
        if not data:
            return table
            
        # 4. Fix table headers
        table = self._fix_table_headers(table, context)
        
        # 5. Detect and handle merged cells
        table = self._handle_merged_cells(table, context)
        
        # 6. Fix data types based on column patterns
        table = self._standardize_data_types(table)
        
        # 7. Apply advanced formatting with Gemini if available
        if self.use_gemini_api:
            table = self._apply_gemini_enhancements(table, context)
        
        # 8. Make a final alignment pass
        table = self._fix_table_alignment(table, context)
        
        return table
        
    def _infer_column_structure(self, table: Dict[str, Any], context: Dict[str, str]) -> Dict[str, Any]:
        """
        Attempt to infer the correct column structure from surrounding text.
        
        Args:
            table: Table dictionary
            context: Text context from surrounding pages
            
        Returns:
            Table with improved column structure
        """
        data = table.get("data", [])
        if not data:
            return table
            
        # Look for column names in surrounding text
        current_page = context.get("current_page", "")
        
        # Extract potential column names from text
        column_patterns = [
            r'columns?\s+(?:include|includes|are|is|lists?|contains?|shows?):\s*([^\n\.]+)',
            r'table\s+\d+[\.\:][^\n\.]*(?:columns?|fields?)[^\n\.]*:\s*([^\n\.]+)',
            r'with\s+columns?\s+for\s+([^\n\.]+)'
        ]
        
        potential_columns = []
        
        for pattern in column_patterns:
            matches = re.finditer(pattern, current_page, re.IGNORECASE)
            for match in matches:
                columns_text = match.group(1).strip()
                # Split by common delimiters
                for delimiter in [',', ';', 'and']:
                    if delimiter in columns_text:
                        column_list = [c.strip() for c in columns_text.split(delimiter) if c.strip()]
                        if len(column_list) >= 2:  # At least two columns
                            potential_columns = column_list
                            break
                
                if potential_columns:
                    break
            
            if potential_columns:
                break
                
        # If potential column names found, check if they match headers or first row
        if potential_columns:
            # Detect if headers already defined
            has_headers = "headers" in table.get("metadata", {}) and table["metadata"]["headers"]
            
            # Check if potential columns match headers
            if has_headers:
                headers = table["metadata"]["headers"]
                match_score = self._calculate_column_match(potential_columns, headers)
                
                # If low match, consider replacing with potential columns
                if match_score < 0.3 and len(potential_columns) >= len(headers) * 0.5:
                    logger.info(f"Replacing headers with context-inferred columns for table on page {table.get('page', 'unknown')}")
                    table["metadata"]["headers"] = potential_columns
                    table["metadata"]["headers_inferred_from_context"] = True
            
            # If no headers, check if first row matches potential columns
            elif data:
                first_row = data[0]
                match_score = self._calculate_column_match(potential_columns, first_row)
                
                # If high match, use first row as headers
                if match_score > 0.5:
                    if "metadata" not in table:
                        table["metadata"] = {}
                    
                    table["metadata"]["headers"] = first_row
                    table["metadata"]["has_detected_header"] = True
                    # Remove header row from data to avoid duplication
                    table["data"] = data[1:]
                    logger.info(f"Detected header row based on context for table on page {table.get('page', 'unknown')}")
                
                # If low match but potential columns exist, use potential columns
                elif len(potential_columns) >= len(first_row) * 0.5:
                    if "metadata" not in table:
                        table["metadata"] = {}
                    
                    table["metadata"]["headers"] = potential_columns
                    table["metadata"]["headers_inferred_from_context"] = True
                    logger.info(f"Using context-inferred columns as headers for table on page {table.get('page', 'unknown')}")
        
        return table
        
    def _calculate_column_match(self, potential_columns: List[str], existing_columns: List[Any]) -> float:
        """
        Calculate similarity score between potential column names and existing columns.
        
        Args:
            potential_columns: List of potential column names extracted from context
            existing_columns: List of existing column names (headers or first row)
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not potential_columns or not existing_columns:
            return 0.0
        
        # Convert all to lowercase strings for comparison
        potential_cols_lower = [str(col).lower() for col in potential_columns]
        existing_cols_lower = [str(col).lower() for col in existing_columns]
        
        # Count exact matches
        exact_matches = 0
        for p_col in potential_cols_lower:
            if p_col in existing_cols_lower:
                exact_matches += 1
        
        # Count partial matches (substring)
        partial_matches = 0
        for p_col in potential_cols_lower:
            for e_col in existing_cols_lower:
                # Skip exact matches already counted
                if p_col == e_col:
                    continue
                    
                # Check if one is substring of the other
                if p_col in e_col or e_col in p_col:
                    partial_matches += 0.5  # Half weight for partial matches
                    break
        
        # Calculate similarity score
        max_matches = max(len(potential_cols_lower), len(existing_cols_lower))
        if max_matches == 0:
            return 0.0
            
        similarity = (exact_matches + partial_matches) / max_matches
        return min(similarity, 1.0)  # Cap at 1.0
    
    def _calculate_table_similarity(self, table1: Dict[str, Any], table2: Dict[str, Any]) -> float:
        """
        Calculate similarity between two tables based on content.
        
        Args:
            table1: First table dictionary
            table2: Second table dictionary
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        # Get table data
        data1 = table1.get("data", [])
        data2 = table2.get("data", [])
        
        # If either table has no data, they can't be similar
        if not data1 or not data2:
            return 0.0
        
        # If tables have very different sizes, they're likely different
        rows1, rows2 = len(data1), len(data2)
        cols1 = max(len(row) for row in data1) if data1 else 0
        cols2 = max(len(row) for row in data2) if data2 else 0
        
        # If dimensions differ by more than 20%, consider tables different
        if abs(rows1 - rows2) / max(rows1, rows2) > 0.2 or abs(cols1 - cols2) / max(cols1, cols2) > 0.2:
            return 0.0
        
        # Compare cell contents
        # Normalize to same dimensions for comparison
        min_rows = min(rows1, rows2)
        min_cols = min(cols1, cols2)
        
        matching_cells = 0
        total_cells = min_rows * min_cols
        
        for i in range(min_rows):
            row1 = data1[i]
            row2 = data2[i]
            
            for j in range(min(len(row1), len(row2), min_cols)):
                cell1 = str(row1[j]).strip() if row1[j] is not None else ""
                cell2 = str(row2[j]).strip() if row2[j] is not None else ""
                
                # Exact match
                if cell1 == cell2:
                    matching_cells += 1
                # Partial match (one is substring of the other)
                elif cell1 in cell2 or cell2 in cell1:
                    matching_cells += 0.5
        
        # Calculate similarity
        if total_cells == 0:
            return 0.0
            
        return matching_cells / total_cells
    
    def _run_custom_analysis(self, table: Dict[str, Any], context: Dict[str, str], analysis_name: str, custom_prompt: str) -> Dict[str, Any]:
        """
        Run a custom analysis specified in the configuration.
        
        Args:
            table: Table dictionary
            context: Text context from surrounding pages
            analysis_name: Name of the custom analysis
            custom_prompt: Custom prompt template
            
        Returns:
            Result of the custom analysis
        """
        if not self.use_gemini_api:
            return {"error": "Gemini API not enabled"}
            
        # Get table data and headers
        data = table.get("data", [])
        if not data:
            return {"error": "No data to analyze"}
            
        headers = table.get("metadata", {}).get("headers", [])
        title = table.get("metadata", {}).get("title", "")
        
        # Format table for prompt
        table_str = "Table Title: " + (title or "Unknown") + "\n\n"
        
        # Add headers if available
        if headers:
            table_str += "Headers: " + " | ".join(str(h) for h in headers) + "\n\n"
        
        # Add data rows (limit to 15 rows to save tokens)
        table_str += "Data:\n"
        for i, row in enumerate(data[:15]):
            table_str += " | ".join(str(cell) for cell in row) + "\n"
        
        if len(data) > 15:
            table_str += f"... (additional {len(data) - 15} rows not shown)\n"
        
        # Add context if available
        context_str = ""
        if context and "current_page" in context:
            context_str = context["current_page"][:1000]  # Limit context length
        
        try:
            # Try with document_text first (as used in the config)
            formatted_prompt = custom_prompt.format(document_text=table_str)
            logger.info(f"Successfully formatted custom prompt using 'document_text' placeholder")
        except KeyError:
            try:
                # Fall back to table-specific placeholders
                formatted_prompt = custom_prompt.format(
                    table_content=table_str,
                    context=context_str,
                    table_title=title or "Unknown"
                )
                logger.info(f"Successfully formatted custom prompt using table-specific placeholders")
            except KeyError as e:
                logger.error(f"Error formatting custom prompt for {analysis_name}: {str(e)}")
                logger.error(f"Make sure your custom prompt uses {{document_text}} or {{table_content}}, {{context}}, {{table_title}} as placeholders")
                return {"error": f"Failed to format custom prompt: {str(e)}"}
        
        # Create the system prompt
        company_context = f" with expertise in {self.company_name} data" if self.company_name else ""
        system_prompt = f"You are an expert in {analysis_name}{company_context}. Provide your analysis in structured JSON format."
        
        try:
            # Call the Gemini API
            response = self._call_gemini_api(formatted_prompt, system_prompt)
            result = self._parse_json_from_response(response)
            
            logger.info(f"Successfully ran custom analysis '{analysis_name}'")
            
            return result
        except Exception as e:
            logger.error(f"Error running custom analysis '{analysis_name}': {str(e)}")
            return {"error": f"Custom analysis failed: {str(e)}"}
    
    def _standardize_data_types(self, table: Dict[str, Any]) -> Dict[str, Any]:
        """
        Standardize data types in table cells based on column patterns.
        
        Args:
            table: Table dictionary
            
        Returns:
            Table with standardized data types
        """
        data = table.get("data", [])
        if not data:
            return table
        
        # Determine number of columns
        num_cols = max(len(row) for row in data)
        if num_cols == 0:
            return table
        
        # Analyze each column to determine likely data type
        column_types = []
        for col_idx in range(num_cols):
            # Collect non-empty values from this column
            col_values = []
            for row in data:
                if col_idx < len(row) and row[col_idx]:
                    col_values.append(str(row[col_idx]).strip())
            
            # Skip empty columns
            if not col_values:
                column_types.append("string")
                continue
            
            # Count patterns
            num_pattern = 0
            date_pattern = 0
            percent_pattern = 0
            currency_pattern = 0
            
            for val in col_values:
                # Check for numeric pattern (including with commas and decimals)
                if re.match(r'^-?[\d,]+\.?\d*$', val):
                    num_pattern += 1
                
                # Check for percentage pattern
                if re.match(r'^-?[\d,]+\.?\d*\s*%$', val) or val.endswith('%'):
                    percent_pattern += 1
                
                # Check for currency pattern
                if re.match(r'^[$€£¥][\d,]+\.?\d*$', val) or re.match(r'^-?[\d,]+\.?\d*\s*[$€£¥]$', val):
                    currency_pattern += 1
                
                # Check for date pattern
                if re.match(r'^\d{1,4}[/\-\.]\d{1,2}[/\-\.]\d{1,4}$', val) or \
                   re.match(r'^[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{2,4}$', val):
                    date_pattern += 1
            
            # Determine column type based on majority pattern
            total = len(col_values)
            if percent_pattern / total > 0.5:
                column_types.append("percent")
            elif currency_pattern / total > 0.5:
                column_types.append("currency")
            elif date_pattern / total > 0.5:
                column_types.append("date")
            elif num_pattern / total > 0.5:
                column_types.append("number")
            else:
                column_types.append("string")
        
        # Standardize data based on column types
        for row_idx, row in enumerate(data):
            for col_idx, cell in enumerate(row):
                if col_idx >= len(column_types):
                    continue
                    
                if not cell:
                    continue
                    
                cell_str = str(cell).strip()
                col_type = column_types[col_idx]
                
                # Standardize based on type
                if col_type == "number":
                    # Remove commas and convert to number
                    try:
                        clean_val = cell_str.replace(',', '')
                        if '.' in clean_val:
                            data[row_idx][col_idx] = float(clean_val)
                        else:
                            data[row_idx][col_idx] = int(clean_val)
                    except ValueError:
                        # Keep as string if conversion fails
                        pass
                        
                elif col_type == "percent":
                    # Convert percentage to decimal value
                    try:
                        clean_val = cell_str.replace('%', '').replace(',', '').strip()
                        data[row_idx][col_idx] = float(clean_val) / 100
                    except ValueError:
                        # Keep as string if conversion fails
                        pass
                        
                elif col_type == "currency":
                    # Extract numeric value from currency
                    try:
                        clean_val = re.sub(r'[$€£¥,]', '', cell_str).strip()
                        data[row_idx][col_idx] = float(clean_val)
                    except ValueError:
                        # Keep as string if conversion fails
                        pass
        
        # Add data type information to metadata
        if "metadata" not in table:
            table["metadata"] = {}
            
        table["metadata"]["column_types"] = column_types
        table["metadata"]["data_types_standardized"] = True
        
        return table
    
    def _save_enhanced_tables(self, tables: List[Dict[str, Any]], output_path: str) -> None:
        """
        Save enhanced tables to a JSON file.
        
        Args:
            tables: List of enhanced table dictionaries
            output_path: Path to save the output file
        """
        try:
            # Create directory if it doesn't exist
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Prepare output data
            output_data = {
                "tables": tables,
                "metadata": {
                    "count": len(tables),
                    "processed_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "processor_version": "1.0"
                }
            }
            
            # Write to file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Successfully saved {len(tables)} enhanced tables to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving enhanced tables to {output_path}: {str(e)}")
            raise
