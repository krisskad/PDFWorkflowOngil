"""
Image Post-Processing Module using Gemini Flash Lite 2 via OpenRouter to identify charts and extract table data.
This module processes images extracted from PDFs and uses Gemini's vision capabilities
to detect charts and extract structured data from them.
"""

import json
import logging
import os
import time
import base64
from typing import Dict, Any, List, Optional, Union
import re
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
project_root = script_dir.parent
env_path = project_root / '.env'
if env_path.exists():
    load_dotenv(env_path)
    
# Configure logging
logger = logging.getLogger("pdf_extraction.image_post_processor")

class ImagePostProcessor:
    """
    Class for post-processing images extracted from PDF documents using Gemini Flash Lite 2 via OpenRouter.
    Identifies charts and extracts structured data from them.
    """
    
    def __init__(self, config: Dict[str, Any], company_name: Optional[str] = None):
        """
        Initialize the image post-processor with configuration.
        
        Args:
            config: Configuration dictionary with API settings and processing options
            company_name: Optional company name for context-aware processing
        """
        self.config = config
        self.company_name = company_name
        
        # Try to get API key from config, environment variable, or .env file
        self.api_key = (
            config.get("openrouter_api_key") or 
            os.environ.get("OPENROUTER_API_KEY")
        )
        
        if not self.api_key:
            raise ValueError("OpenRouter API key is required. Set it in config, OPENROUTER_API_KEY environment variable, or .env file.")
            
        # Try to get other settings from config or environment variables
        self.model = config.get("gemini_model") or os.environ.get("GEMINI_MODEL", "google/gemini-2.0-flash-lite-001")
        self.max_tokens = int(config.get("max_tokens") or os.environ.get("MAX_TOKENS", "4096"))
        self.temperature = float(config.get("temperature") or os.environ.get("TEMPERATURE", "0.0"))
        self.rate_limit_delay = config.get("rate_limit_delay", 1.0)  # Seconds between API calls
        
        # Chart detection settings
        self.chart_detection_enabled = config.get("chart_detection", {}).get("enabled", True)
        self.chart_confidence_threshold = config.get("chart_detection", {}).get("confidence_threshold", 0.7)
        
        # Table extraction settings
        self.table_extraction_enabled = config.get("table_extraction", {}).get("enabled", True)
        self.include_headers = config.get("table_extraction", {}).get("include_headers", True)
        self.standardize_data_types = config.get("table_extraction", {}).get("standardize_data_types", True)
        
        # Enhanced analysis settings
        self.summarize_charts = config.get("enhanced_analysis", {}).get("summarize_charts", True)
        self.extract_keywords = config.get("enhanced_analysis", {}).get("extract_keywords", True)
        self.extract_insights = config.get("enhanced_analysis", {}).get("extract_insights", True)
        self.detect_trends = config.get("enhanced_analysis", {}).get("detect_trends", True)
        
        # Optional custom prompts
        self.custom_prompts = config.get("custom_prompts", {})
        
        # Site information for OpenRouter
        self.site_url = config.get("site_url") or os.environ.get("SITE_URL", "https://example.com")
        self.site_name = config.get("site_name") or os.environ.get("SITE_NAME", "PDF Extraction Tool")
        
        # Initialize OpenAI client with OpenRouter configuration
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )
        
        logger.info(f"Initialized Image Post-Processor with model: {self.model}")
        if self.company_name:
            logger.info(f"Processing images for company: {self.company_name}")
    
    def _get_media_type(self, image_path: str) -> str:
        """
        Determine the media type based on the image file extension.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            MIME type string for the image
        """
        ext = os.path.splitext(image_path.lower())[1]
        
        if ext == '.png':
            return 'image/png'
        elif ext in ['.jpg', '.jpeg']:
            return 'image/jpeg'
        elif ext == '.gif':
            return 'image/gif'
        elif ext == '.webp':
            return 'image/webp'
        elif ext == '.bmp':
            return 'image/bmp'
        elif ext == '.svg':
            return 'image/svg'
        else:
            # Default to PNG if unknown
            logger.warning(f"Unknown image extension {ext}, defaulting to image/png")
            return 'image/png'
        
    def process_images(self, images_data: List[Dict[str, Any]], output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a list of images to identify charts and extract table data.
        
        Args:
            images_data: List of image metadata dictionaries (from ImageExtractor)
            output_path: Optional path to save the processed results
            
        Returns:
            Dictionary containing processed image data with chart detection and table extraction
        """
        logger.info(f"Starting image post-processing for {len(images_data)} images")
        
        processed_images = []
        
        for image_data in images_data:
            image_path = image_data.get("path")
            
            if not image_path or not os.path.exists(image_path):
                logger.warning(f"Image path does not exist: {image_path}")
                continue
                
            # Skip SVG files as they're vector graphics and not suitable for chart detection
            if image_path.lower().endswith('.svg'):
                logger.info(f"Skipping SVG file: {image_path}")
                continue
                
            logger.info(f"Processing image: {image_path}")
            
            try:
                # Process the image
                processed_image = self._process_single_image(image_data)
                processed_images.append(processed_image)
                
                # Save incremental results if output path is provided
                if output_path:
                    self._save_processed_data({"processed_images": processed_images}, output_path)
                    
            except Exception as e:
                logger.error(f"Error processing image {image_path}: {str(e)}")
                # Add the original image data with an error flag
                processed_images.append({
                    **image_data,
                    "error": str(e),
                    "is_chart": False,
                    "processed": False
                })
        
        # Prepare final output
        result = {
            "processed_images": processed_images,
            "summary": {
                "total_images": len(images_data),
                "processed_images": len(processed_images),
                "charts_detected": sum(1 for img in processed_images if img.get("is_chart", False)),
                "tables_extracted": sum(1 for img in processed_images if "extracted_table" in img)
            }
        }
        
        # Save final results if output path is provided
        if output_path:
            self._save_processed_data(result, output_path)
            logger.info(f"Saved final processed data to {output_path}")
        
        return result
    
    def _process_single_image(self, image_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single image to detect if it's a chart and extract table data if applicable.
        
        Args:
            image_data: Dictionary containing image metadata
            
        Returns:
            Dictionary with the original metadata plus chart detection and table extraction results
        """
        image_path = image_data.get("path")
        
        # Create a copy of the original image data
        processed_image = image_data.copy()
        processed_image["processed"] = True
        
        # Detect if the image is a chart
        if self.chart_detection_enabled:
            chart_detection = self._detect_chart(image_path)
            processed_image.update(chart_detection)
            
            # If it's a chart and table extraction is enabled, extract the table data
            if chart_detection.get("is_chart", False) and self.table_extraction_enabled:
                table_data = self._extract_table_from_chart(image_path, chart_detection.get("chart_type"))
                processed_image["extracted_table"] = table_data
                    
                # Apply enhanced analysis for charts
                if chart_detection.get("is_chart", False):
                    # Initialize AI enhancements if not present
                    if "ai_enhancements" not in processed_image:
                        processed_image["ai_enhancements"] = {}
                    
                    # Check which enhancements are enabled
                    enhancements_needed = []
                    if self.summarize_charts:
                        enhancements_needed.append("summary")
                    if self.extract_keywords:
                        enhancements_needed.append("keywords")
                    if self.extract_insights:
                        enhancements_needed.append("insights")
                    if self.detect_trends:
                        enhancements_needed.append("trends")
                    
                    # If any enhancements are needed, make a single API call
                    if enhancements_needed:
                        enhancements = self._get_combined_chart_enhancements(image_path, chart_detection, table_data, enhancements_needed)
                        
                        # Add each enhancement to the processed image
                        for enhancement_type, enhancement_data in enhancements.items():
                            processed_image["ai_enhancements"][enhancement_type] = enhancement_data
                        
                        logger.info(f"Applied AI enhancements ({', '.join(enhancements_needed)}) for chart: {image_path} with a single API call")
                    
                    # Apply any custom analysis defined in the config
                    if self.custom_prompts:
                        logger.info(f"Found {len(self.custom_prompts)} custom prompts to apply: {', '.join(self.custom_prompts.keys())}")
                        
                        # Initialize AI enhancements if not present
                        if "ai_enhancements" not in processed_image:
                            processed_image["ai_enhancements"] = {}
                        
                        # Apply each custom prompt
                        for analysis_name, prompt in self.custom_prompts.items():
                            logger.info(f"Applying custom analysis '{analysis_name}' for chart: {image_path}")
                            custom_analysis = self._run_custom_analysis(image_path, chart_detection, table_data, analysis_name, prompt)
                            processed_image["ai_enhancements"][analysis_name] = custom_analysis
                            logger.info(f"Applied custom analysis '{analysis_name}' for chart: {image_path}")
                    else:
                        logger.info("No custom prompts found in configuration")
        
        return processed_image
    
    def _detect_chart(self, image_path: str) -> Dict[str, Any]:
        """
        Detect if an image contains a chart using Gemini's vision capabilities.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with chart detection results
        """
        logger.info(f"Detecting if image is a chart: {image_path}")
        
        # Encode the image as base64
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        
        # Create the prompt
        company_context = f" specializing in {self.company_name} data" if self.company_name else ""
        prompt = f"""
        You are an expert in chart and graph detection and analysis{company_context}, with particular expertise in identifying 
        demographic data visualizations, statistical charts, and business reporting visuals. Your task is to determine if an image 
        contains a chart or graph, and if so, identify the type of chart and provide details about it. 
        
        Pay special attention to:
        - Bar charts and column charts showing demographic or statistical information
        - Charts with multiple data series or time periods (e.g., years 20, 21, 22)
        - Visualizations showing percentages, trends, or comparisons across categories
        - Charts that may appear in corporate reports, ESG documents, or diversity reports
        
        Analyze this image and determine if it contains a chart or graph.
        
        Pay special attention to bar charts, column charts, and data visualizations showing demographic or statistical information. 
        Look for patterns of bars or columns arranged to show comparisons between different categories or time periods.
        
        If it is a chart or graph, provide the following information in JSON format:
        1. "is_chart": true
        2. "chart_type": The specific type of chart (e.g., bar chart, column chart, line graph, pie chart, scatter plot, etc.)
        3. "confidence": Your confidence level (0-1) that this is a chart
        4. "description": A brief description of what the chart shows
        5. "axes_labels": The labels for x and y axes if applicable
        6. "data_series": Names of data series shown in the chart if identifiable
        
        If it is NOT a chart or graph, respond with:
        {{
          "is_chart": false,
          "confidence": (your confidence level that this is NOT a chart),
          "content_type": (what the image appears to contain instead)
        }}
        
        Respond with ONLY the JSON object. No other text.
        """
        
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
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{self._get_media_type(image_path)};base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            # Add a delay to respect rate limits
            time.sleep(self.rate_limit_delay)
            
            # Extract the JSON response
            result = self._parse_json_from_response(response)
            
            # Apply confidence threshold
            if result.get("is_chart", False) and result.get("confidence", 0) < self.chart_confidence_threshold:
                result["is_chart"] = False
                result["below_threshold"] = True
                logger.info(f"Chart detection confidence {result.get('confidence', 0)} below threshold {self.chart_confidence_threshold}")
            
            logger.info(f"Chart detection result: is_chart={result.get('is_chart', False)}, " 
                       f"type={result.get('chart_type', 'N/A')}, confidence={result.get('confidence', 0)}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in chart detection: {str(e)}")
            return {"is_chart": False, "error": str(e)}
    
    def _generate_chart_summary(self, image_path: str, chart_detection: Dict[str, Any], table_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of the chart's content and meaning.
        
        Args:
            image_path: Path to the chart image
            chart_detection: Chart detection results
            table_data: Extracted table data from the chart
            
        Returns:
            Dictionary with chart summary information
        """
        logger.info(f"Generating summary for chart: {image_path}")
        
        # Encode the image as base64
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        
        # Get chart type and description from detection results
        chart_type = chart_detection.get("chart_type", "Unknown")
        chart_description = chart_detection.get("description", "")
        
        # Format table data if available
        table_str = ""
        if table_data and "data" in table_data:
            headers = table_data.get("headers", [])
            data = table_data.get("data", [])
            
            if headers:
                table_str += "Headers: " + " | ".join(str(h) for h in headers) + "\n\n"
            
            table_str += "Data:\n"
            for row in data[:10]:  # Limit to 10 rows
                table_str += " | ".join(str(cell) for cell in row) + "\n"
            
            if len(data) > 10:
                table_str += "... (additional rows not shown)\n"
        
        # Create the prompt
        company_context = f" with expertise in {self.company_name} data" if self.company_name else ""
        company_info = f" from {self.company_name}" if self.company_name else ""
        prompt = f"""
        You are an expert in data visualization and chart analysis{company_context}. Your task is to summarize and explain
        charts in a clear, concise manner. Provide your summary in structured JSON format.
        
        Generate a comprehensive summary of this {chart_type} chart{company_info}.
        
        Chart description: {chart_description}
        
        Extracted data:
        {table_str}
        
        For your response, provide the following in JSON format:
        1. "brief_summary": A 1-2 sentence summary of what this chart shows
        2. "detailed_summary": A more comprehensive explanation (3-5 sentences)
        3. "key_observations": An array of 3-5 key observations from the chart
        4. "data_range": The range or scope of data presented (e.g., time period, categories)
        5. "audience": Who would find this chart most useful
        6. "purpose": The likely purpose of this chart in the document
        
        Respond with ONLY a JSON object containing your summary. No other text.
        """
        
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
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{self._get_media_type(image_path)};base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            # Add a delay to respect rate limits
            time.sleep(self.rate_limit_delay)
            
            # Extract the JSON response
            result = self._parse_json_from_response(response)
            
            logger.info(f"Successfully generated summary for chart: {image_path}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating chart summary: {str(e)}")
            return {"error": str(e)}
    
    def _extract_chart_keywords(self, image_path: str, chart_detection: Dict[str, Any], table_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract keywords and topics from the chart.
        
        Args:
            image_path: Path to the chart image
            chart_detection: Chart detection results
            table_data: Extracted table data from the chart
            
        Returns:
            Dictionary with keywords and topics
        """
        logger.info(f"Extracting keywords for chart: {image_path}")
        
        # Encode the image as base64
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        
        # Get chart type and description from detection results
        chart_type = chart_detection.get("chart_type", "Unknown")
        
        # Create the prompt
        company_context = f" with expertise in {self.company_name} data" if self.company_name else ""
        company_info = f" from {self.company_name}" if self.company_name else ""
        prompt = f"""
        You are an expert in data visualization and topic extraction{company_context}. Your task is to identify the main topics,
        themes, and keywords in charts and graphs. Provide your analysis in structured JSON format.
        
        Extract the main topics and keywords from this {chart_type} chart{company_info}.
        
        For your response, provide the following in JSON format:
        1. "main_topics": An array of 3-5 main topics covered in the chart
        2. "keywords": An array of 5-10 important keywords from the chart
        3. "categories": The main categories or dimensions of data presented
        4. "entities": Important named entities shown (people, organizations, products, etc.)
        5. "industry_relevance": Industries or sectors this chart is most relevant to
        
        Respond with ONLY a JSON object containing your keyword analysis. No other text.
        """
        
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
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{self._get_media_type(image_path)};base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            # Add a delay to respect rate limits
            time.sleep(self.rate_limit_delay)
            
            # Extract the JSON response
            result = self._parse_json_from_response(response)
            
            logger.info(f"Successfully extracted keywords for chart: {image_path}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error extracting chart keywords: {str(e)}")
            return {"error": str(e)}
    
    
    def _extract_chart_insights(self, image_path: str, chart_detection: Dict[str, Any], table_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract key insights from the chart.
        
        Args:
            image_path: Path to the chart image
            chart_detection: Chart detection results
            table_data: Extracted table data from the chart
            
        Returns:
            Dictionary with insights and analysis
        """
        logger.info(f"Extracting insights for chart: {image_path}")
        
        # Encode the image as base64
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        
        # Get chart type from detection results
        chart_type = chart_detection.get("chart_type", "Unknown")
        
        # Create the prompt
        company_context = f" with expertise in {self.company_name} data" if self.company_name else ""
        company_info = f" from {self.company_name}" if self.company_name else ""
        prompt = f"""
        You are an expert in data analysis and visualization{company_context}. Your task is to extract key insights,
        implications, and business value from charts and graphs. Provide your analysis in structured JSON format.
        
        Analyze this {chart_type} chart{company_info} and extract key insights and business implications.
        
        For your response, provide the following in JSON format:
        1. "key_insights": An array of 3-5 important insights from the chart
        2. "business_implications": What these findings mean for business decisions
        3. "recommendations": Potential actions based on the chart data
        4. "limitations": Any limitations or caveats about the data presentation
        5. "comparative_analysis": How different elements in the chart compare to each other
        
        Respond with ONLY a JSON object containing your insights. No other text.
        """
        
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
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{self._get_media_type(image_path)};base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            # Add a delay to respect rate limits
            time.sleep(self.rate_limit_delay)
            
            # Extract the JSON response
            result = self._parse_json_from_response(response)
            
            logger.info(f"Successfully extracted insights for chart: {image_path}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error extracting chart insights: {str(e)}")
            return {"error": str(e)}
    
    def _get_combined_chart_enhancements(self, image_path: str, chart_detection: Dict[str, Any], table_data: Dict[str, Any], enhancements_needed: List[str]) -> Dict[str, Any]:
        """
        Make a single API call to get all requested chart enhancements.
        
        Args:
            image_path: Path to the chart image
            chart_detection: Chart detection results
            table_data: Extracted table data from the chart
            enhancements_needed: List of enhancement types to request
            
        Returns:
            Dictionary with all requested enhancements
        """
        logger.info(f"Getting combined chart enhancements for: {image_path}")
        
        if not enhancements_needed:
            return {}
            
        # Encode the image as base64
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        
        # Get chart type and description from detection results
        chart_type = chart_detection.get("chart_type", "Unknown")
        chart_description = chart_detection.get("description", "")
        
        # Format table data if available
        table_str = ""
        if table_data and "data" in table_data:
            headers = table_data.get("headers", [])
            data = table_data.get("data", [])
            
            if headers:
                table_str += "Headers: " + " | ".join(str(h) for h in headers) + "\n\n"
            
            table_str += "Data:\n"
            for row in data[:10]:  # Limit to 10 rows
                table_str += " | ".join(str(cell) for cell in row) + "\n"
            
            if len(data) > 10:
                table_str += f"... (additional {len(data) - 10} rows not shown)\n"
        
        # Build the prompt based on requested enhancements
        company_context = f" with expertise in {self.company_name} data" if self.company_name else ""
        company_info = f" from {self.company_name}" if self.company_name else ""
        prompt = f"""
        You are an expert in data visualization, chart analysis, and business intelligence{company_context}. Your task is to analyze charts
        and provide multiple types of analysis in a single response. Structure your response as a JSON object
        with separate sections for each type of analysis requested.
        
        Analyze this {chart_type} chart{company_info} and provide the requested information.
        
        Chart description: {chart_description}
        
        Extracted data:
        {table_str}
        
        Please provide the following analyses in a SINGLE JSON object:
        """
        
        # Add specific requests for each enhancement type
        if "summary" in enhancements_needed:
            prompt += """
            "summary": {
                "brief_summary": A 1-2 sentence summary of what this chart shows,
                "detailed_summary": A more comprehensive explanation (3-5 sentences),
                "key_observations": An array of 3-5 key observations from the chart,
                "data_range": The range or scope of data presented (e.g., time period, categories),
                "audience": Who would find this chart most useful,
                "purpose": The likely purpose of this chart in the document
            },
            """
        
        if "keywords" in enhancements_needed:
            prompt += """
            "keywords": {
                "main_topics": An array of 3-5 main topics covered in the chart,
                "keywords": An array of 5-10 important keywords from the chart,
                "categories": The main categories or dimensions of data presented,
                "entities": Important named entities shown (people, organizations, products, etc.),
                "industry_relevance": Industries or sectors this chart is most relevant to
            },
            """
        
        if "insights" in enhancements_needed:
            prompt += """
            "insights": {
                "key_insights": An array of 3-5 important insights from the chart,
                "business_implications": What these findings mean for business decisions,
                "recommendations": Potential actions based on the chart data,
                "limitations": Any limitations or caveats about the data presentation,
                "comparative_analysis": How different elements in the chart compare to each other
            },
            """
        
        if "trends" in enhancements_needed:
            prompt += """
            "trends": {
                "trends": An array of trends visible in the data,
                "patterns": Recurring patterns or cycles in the data,
                "outliers": Any notable outliers or anomalies,
                "correlations": Apparent correlations between variables,
                "growth_rates": Growth or decline rates if applicable,
                "statistical_significance": Assessment of whether patterns appear statistically significant
            }
            """
        
        prompt += """
        
        Respond with ONLY a JSON object containing the requested analyses. No other text.
        """
        
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
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{self._get_media_type(image_path)};base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            # Add a delay to respect rate limits
            time.sleep(self.rate_limit_delay)
            
            # Extract the JSON response
            result = self._parse_json_from_response(response)
            
            # Ensure all requested enhancements are present in the result
            for enhancement in enhancements_needed:
                if enhancement not in result:
                    result[enhancement] = {"error": f"Failed to generate {enhancement}"}
            
            logger.info(f"Successfully generated combined enhancements for chart: {image_path}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating combined chart enhancements: {str(e)}")
            return {enhancement: {"error": str(e)} for enhancement in enhancements_needed}
    
    def _run_custom_analysis(self, image_path: str, chart_detection: Dict[str, Any], table_data: Dict[str, Any], analysis_name: str, custom_prompt: str) -> Dict[str, Any]:
        """
        Run a custom analysis specified in the configuration.
        
        Args:
            image_path: Path to the chart image
            chart_detection: Chart detection results
            table_data: Extracted table data from the chart
            analysis_name: Name of the custom analysis
            custom_prompt: Custom prompt template
            
        Returns:
            Result of the custom analysis
        """
        logger.info(f"Running custom analysis '{analysis_name}' for chart: {image_path}")
        
        # Encode the image as base64
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        
        # Get chart type and description from detection results
        chart_type = chart_detection.get("chart_type", "Unknown")
        chart_description = chart_detection.get("description", "")
        
        # Format table data if available
        table_str = ""
        if table_data and "data" in table_data:
            headers = table_data.get("headers", [])
            data = table_data.get("data", [])
            
            if headers:
                table_str += "Headers: " + " | ".join(str(h) for h in headers) + "\n\n"
            
            table_str += "Data:\n"
            for row in data[:10]:  # Limit to 10 rows
                table_str += " | ".join(str(cell) for cell in row) + "\n"
            
            if len(data) > 10:
                table_str += f"... (additional {len(data) - 10} rows not shown)\n"
        
        # Prepare the prompt by inserting the chart information
        try:
            # Try with document_text first (as used in the config)
            formatted_prompt = custom_prompt.format(document_text=table_str)
            logger.info(f"Successfully formatted custom prompt using 'document_text' placeholder")
        except KeyError:
            try:
                # Fall back to chart-specific placeholders
                formatted_prompt = custom_prompt.format(
                    chart_type=chart_type,
                    chart_description=chart_description,
                    table_data=table_str,
                    image_path=os.path.basename(image_path)
                )
                logger.info(f"Successfully formatted custom prompt using chart-specific placeholders")
            except KeyError:
                # If all formatting attempts fail, use the custom prompt as is
                logger.warning(f"Could not format custom prompt with any known placeholders. Using as is.")
                formatted_prompt = custom_prompt
        
        # Create the prompt with context
        company_context = f" with expertise in {self.company_name} data" if self.company_name else ""
        prompt = f"You are an expert in {analysis_name}{company_context}. Provide your analysis in structured JSON format.\n\n{formatted_prompt}"
        
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
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{self._get_media_type(image_path)};base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            # Add a delay to respect rate limits
            time.sleep(self.rate_limit_delay)
            
            # Extract the JSON response
            result = self._parse_json_from_response(response)
            
            logger.info(f"Successfully ran custom analysis '{analysis_name}' for chart: {image_path}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error running custom analysis '{analysis_name}': {str(e)}")
            return {"error": f"Custom analysis failed: {str(e)}"}
    
    def _detect_chart_trends(self, image_path: str, chart_detection: Dict[str, Any], table_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect trends and patterns in the chart data.
        
        Args:
            image_path: Path to the chart image
            chart_detection: Chart detection results
            table_data: Extracted table data from the chart
            
        Returns:
            Dictionary with trends and patterns
        """
        logger.info(f"Detecting trends for chart: {image_path}")
        
        # Encode the image as base64
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        
        # Get chart type from detection results
        chart_type = chart_detection.get("chart_type", "Unknown")
        
        # Format table data if available
        table_str = ""
        if table_data and "data" in table_data:
            headers = table_data.get("headers", [])
            data = table_data.get("data", [])
            
            if headers:
                table_str += "Headers: " + " | ".join(str(h) for h in headers) + "\n\n"
            
            table_str += "Data:\n"
            for row in data:
                table_str += " | ".join(str(cell) for cell in row) + "\n"
        
        # Create the prompt
        company_context = f" with expertise in {self.company_name} data" if self.company_name else ""
        company_info = f" from {self.company_name}" if self.company_name else ""
        prompt = f"""
        You are an expert in data analysis and pattern recognition{company_context}. Your task is to identify trends,
        patterns, and statistical insights in charts and graphs. Provide your analysis in structured JSON format.
        
        Analyze this {chart_type} chart{company_info} and identify trends, patterns, and statistical insights.
        
        Extracted data:
        {table_str}
        
        For your response, provide the following in JSON format:
        1. "trends": An array of trends visible in the data
        2. "patterns": Recurring patterns or cycles in the data
        3. "outliers": Any notable outliers or anomalies
        4. "correlations": Apparent correlations between variables
        5. "growth_rates": Growth or decline rates if applicable
        6. "statistical_significance": Assessment of whether patterns appear statistically significant
        
        Respond with ONLY a JSON object containing your trend analysis. No other text.
        """
        
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
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{self._get_media_type(image_path)};base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            # Add a delay to respect rate limits
            time.sleep(self.rate_limit_delay)
            
            # Extract the JSON response
            result = self._parse_json_from_response(response)
            
            logger.info(f"Successfully detected trends for chart: {image_path}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error detecting chart trends: {str(e)}")
            return {"error": str(e)}
    
    def _extract_table_from_chart(self, image_path: str, chart_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract tabular data from a chart image using Gemini's vision capabilities.
        
        Args:
            image_path: Path to the chart image
            chart_type: Optional type of chart to guide the extraction
            
        Returns:
            Dictionary with extracted table data
        """
        logger.info(f"Extracting table data from chart: {image_path}")
        
        # Encode the image as base64
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        
        # Create the prompt
        company_context = f" with expertise in {self.company_name} data" if self.company_name else ""
        chart_type_info = f"This is a {chart_type}. " if chart_type else ""
        prompt = f"""
        You are an expert in data extraction from charts and graphs{company_context}. Your task is to extract the underlying tabular data
        from a chart image with high precision. Respond with structured JSON only.
        
        {chart_type_info}Extract the underlying data from this chart into a structured table format.
        
        Pay special attention to:
        - Demographic data visualizations with multiple categories (gender, ethnicity, etc.)
        - Charts showing data across multiple time periods (years, quarters)
        - Percentage values and their proper attribution to categories
        - Both the overall data and any breakdowns by subcategory
        
        Provide the following in your response:
        1. "headers": Column headers for the data table
        2. "data": The extracted data as a 2D array (rows and columns)
        3. "data_types": The data type of each column (string, number, date, etc.)
        4. "units": The units for each column if applicable (%, count, ratio)
        5. "title": The title of the chart if present
        6. "source": The data source if mentioned in the chart
        7. "notes": Any footnotes or additional information
        8. "extraction_confidence": Your confidence in the accuracy of the extraction (0-1)
        
        Be as precise as possible with numeric values. If you can't determine an exact value, provide your best estimate.
        For demographic data, ensure you capture all categories and time periods shown.
        
        Respond with ONLY the JSON object. No other text.
        """
        
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
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{self._get_media_type(image_path)};base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            # Add a delay to respect rate limits
            time.sleep(self.rate_limit_delay)
            
            # Extract the JSON response
            result = self._parse_json_from_response(response)
            
            # Standardize data types if enabled
            if self.standardize_data_types and "data" in result and "data_types" in result:
                result["data"] = self._standardize_data_values(result["data"], result["data_types"])
            
            logger.info(f"Successfully extracted table data from chart with confidence: {result.get('extraction_confidence', 0)}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in table extraction: {str(e)}")
            return {"error": str(e)}
    
    def _standardize_data_values(self, data: List[List[Any]], data_types: List[str]) -> List[List[Any]]:
        """
        Standardize data values based on their data types.
        
        Args:
            data: 2D array of data values
            data_types: List of data types for each column
            
        Returns:
            2D array with standardized data values
        """
        standardized_data = []
        
        for row in data:
            standardized_row = []
            
            for i, value in enumerate(row):
                if i < len(data_types):
                    data_type = data_types[i].lower()
                    
                    if data_type in ["number", "float", "integer", "decimal"]:
                        try:
                            # Try to convert to float or int
                            if isinstance(value, str):
                                # Remove commas and other formatting
                                clean_value = value.replace(",", "").replace("$", "").replace("%", "").strip()
                                if "." in clean_value:
                                    standardized_row.append(float(clean_value))
                                else:
                                    standardized_row.append(int(clean_value))
                            else:
                                standardized_row.append(value)
                        except (ValueError, TypeError):
                            standardized_row.append(value)
                    else:
                        standardized_row.append(value)
                else:
                    standardized_row.append(value)
            
            standardized_data.append(standardized_row)
        
        return standardized_data
    
    def _parse_json_from_response(self, response) -> Dict[str, Any]:
        """
        Extract and parse JSON from Gemini's response via OpenRouter.
        
        Args:
            response: OpenAI API response from OpenRouter
            
        Returns:
            Parsed JSON data
        """
        try:
            # Extract text from the response
            text = response.choices[0].message.content if response.choices else ""
            
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
    
    def _save_processed_data(self, data: Dict[str, Any], output_path: str) -> None:
        """
        Save the processed data to a JSON file.
        
        Args:
            data: The processed image data
            output_path: Path to save the output
        """
        try:
            # Ensure the output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Failed to save processed data: {str(e)}")
