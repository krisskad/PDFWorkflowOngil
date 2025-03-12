"""
Text Post-Processing Module using Gemini Flash Lite 2 via OpenRouter to enhance PDF extraction results.
This script processes the text output from PDFParser and applies advanced text analysis.
"""

import json
import logging
import os
import time
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
logger = logging.getLogger("pdf_extraction.text_post_processor")

class TextPostProcessor:
    """
    Class for post-processing PDF extraction text results using Gemini Flash Lite 2 via OpenRouter.
    Enhances the extracted text content with AI-powered analysis.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the text post-processor with configuration.
        
        Args:
            config: Configuration dictionary with API settings and processing options
        """
        self.config = config
        
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
        self.extract_entities = config.get("extract_entities", True)
        self.extract_sections = config.get("extract_sections", True)
        self.summarize_content = config.get("summarize_content", True)
        self.detect_document_type = config.get("detect_document_type", True)
        self.extract_key_information = config.get("extract_key_information", True)
        self.extract_topics = config.get("extract_topics", True)
        self.rate_limit_delay = config.get("rate_limit_delay", 1.0)  # Seconds between API calls
        self.company_name = config.get("company_name")  # Company name for targeted analysis
        
        # Site information for OpenRouter
        self.site_url = config.get("site_url") or os.environ.get("SITE_URL", "https://example.com")
        self.site_name = config.get("site_name") or os.environ.get("SITE_NAME", "PDF Extraction Tool")
        
        # Initialize OpenAI client with OpenRouter configuration
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )
        
        # Optional custom prompts
        self.custom_prompts = config.get("custom_prompts", {})
        
        logger.info(f"Initialized Text Post-Processor with model: {self.model}")
        
    def process_text_content(self, parser_output: Dict[str, Any], output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Process the text content using Gemini API for enhanced analysis.
        Updates the enhanced output after each API call and saves incrementally.
        
        Args:
            parser_output: Output from PDFParser.extract_all()
            output_path: Optional path to save the enhanced output
            
        Returns:
            Enhanced text content with AI-powered analysis
        """
        logger.info("Starting Gemini post-processing of text content")
        
        # Create a new dictionary with the original content plus AI enhancements
        enhanced_output = {
            "original_content": parser_output,
            "ai_enhancements": {}
        }
        
        # Extract the full text content for document-level analysis
        text_content = self._extract_text_content(parser_output)
        
        # Function to save the current state of enhanced output
        def save_current_state():
            if output_path:
                self._save_enhanced_output(enhanced_output, output_path)
                logger.info(f"Saved current state of enhanced output to {output_path}")
        
        # Detect document type (document-level)
        if self.detect_document_type:
            document_type = self._detect_document_type(text_content)
            enhanced_output["ai_enhancements"]["document_type"] = document_type
            doc_type = document_type.get("type", "unknown")
            logger.info(f"Detected document type: {doc_type}")
            save_current_state()  # Save after document type detection
        else:
            doc_type = "unknown"
        
        # Extract document sections and structure
        if self.extract_sections:
            sections = self._extract_sections(text_content)
            enhanced_output["ai_enhancements"]["document_structure"] = sections
            logger.info(f"Extracted document structure with {len(sections)} sections")
            save_current_state()  # Save after section extraction
        
        # Create content summary (document-level)
        if self.summarize_content:
            summary = self._generate_summary(text_content)
            enhanced_output["ai_enhancements"]["summary"] = summary
            logger.info("Generated document summary")
            save_current_state()  # Save after summary generation
        
        # Extract document-level topics
        if self.extract_topics:
            topics = self._extract_topics(text_content, is_page_level=False)
            enhanced_output["ai_enhancements"]["topics"] = topics
            logger.info(f"Extracted document-level topics")
            save_current_state()  # Save after document-level topic extraction
        
        # Extract document-level entities
        if self.extract_entities:
            entities = self._extract_entities(text_content, doc_type, is_page_level=False)
            enhanced_output["ai_enhancements"]["entities"] = entities
            logger.info(f"Extracted document-level entities")
            save_current_state()  # Save after document-level entity extraction
        
        # Extract key information based on document type (document-level)
        if self.extract_key_information:
            key_info = self._extract_key_information(text_content, doc_type)
            enhanced_output["ai_enhancements"]["key_information"] = key_info
            logger.info(f"Extracted key information for document type: {doc_type}")
            save_current_state()  # Save after key information extraction
        
        # Process each page individually for page-level analysis
        if "text" in parser_output and "pages" in parser_output["text"]:
            pages = parser_output["text"]["pages"]
            page_analyses = []
            
            for page in pages:
                if "content" in page and page["content"].strip():
                    page_num = page.get("page_number", "unknown")
                    page_content = page["content"]
                    page_title = self._extract_page_title(page_content)
                    
                    logger.info(f"Processing page {page_num}: {page_title}")
                    
                    page_analysis = {
                        "page_number": page_num,
                        "title": page_title
                    }
                    
                    # Extract topics for this page
                    if self.extract_topics:
                        page_topics = self._extract_topics(page_content, is_page_level=True)
                        page_analysis["topics"] = page_topics
                    
                    # Extract entities for this page based on document type
                    if self.extract_entities:
                        page_entities = self._extract_entities(page_content, doc_type, is_page_level=True)
                        page_analysis["entities"] = page_entities
                    
                    page_analyses.append(page_analysis)
                    
                    # Update the enhanced output with the current page analysis
                    enhanced_output["ai_enhancements"]["page_level_analysis"] = page_analyses
                    logger.info(f"Completed analysis for page {page_num}")
                    save_current_state()  # Save after each page analysis
            
            logger.info(f"Completed page-level analysis for {len(page_analyses)} pages")
        
        # Apply any custom analysis defined in the config (document-level)
        if self.custom_prompts:
            logger.info(f"Found {len(self.custom_prompts)} custom prompts to apply: {', '.join(self.custom_prompts.keys())}")
            for analysis_name, prompt in self.custom_prompts.items():
                logger.info(f"Applying custom analysis: {analysis_name}")
                custom_analysis = self._run_custom_analysis(text_content, analysis_name, prompt)
                enhanced_output["ai_enhancements"][analysis_name] = custom_analysis
                logger.info(f"Applied custom analysis: {analysis_name}")
                save_current_state()  # Save after each custom analysis
        else:
            logger.info("No custom prompts found in configuration")
        
        # Final save of enhanced output
        if output_path:
            self._save_enhanced_output(enhanced_output, output_path)
            logger.info(f"Saved final enhanced output to {output_path}")
        
        return enhanced_output
        
    def _extract_page_title(self, page_content: str) -> str:
        """
        Extract the title or heading from a page's content using Gemini API.
        
        Args:
            page_content: Text content from a single page
            
        Returns:
            Extracted title or "Untitled Page" if none found
        """
        # Use Gemini API to extract a meaningful title
        try:
            # Prepare a short excerpt of the page content for the prompt
            # Limit to first 1000 characters to save tokens
            content_excerpt = page_content[:1000]
            
            # Create the system prompt
            system_prompt = """
            You are an expert in document analysis and information extraction.
            Your task is to identify the most appropriate title for a page of text.
            Provide ONLY the title, with no additional text, explanation, or formatting.
            """
            
            # Add company context if available
            company_context = ""
            if self.company_name:
                company_context = f"This document is related to the company '{self.company_name}'. Consider this when extracting the title."
            
            # Create the user prompt
            prompt = f"""
            Extract the most appropriate title for this page of text.
            The title should be concise (typically 3-10 words) but descriptive of the page's main content.
            {company_context}
            
            If there's a clear heading or title at the beginning of the text, use that.
            If there's no clear title, generate an appropriate descriptive title based on the content.
            
            Page content:
            ```
            {content_excerpt}
            ```
            
            Respond with ONLY the title text. No other text, no quotes, no explanations.
            """
            
            # Call the Gemini API via OpenRouter
            response = self._call_gemini_api(prompt, system_prompt)
            
            # Extract the title from the response
            if response and "content" in response and response["content"]:
                title = response["content"][0].get("text", "").strip()
                
                # If the title is too long, truncate it
                if len(title) > 100:
                    title = title[:97] + "..."
                
                # If we got a valid title, return it
                if title:
                    return title
            
            # Fall back to the simple approach if API call fails or returns empty result
            logger.info("Falling back to simple title extraction method")
            
        except Exception as e:
            logger.error(f"Error extracting title with Gemini API: {str(e)}")
            logger.info("Falling back to simple title extraction method")
        
        # Simple fallback approach
        lines = page_content.split('\n')
        for line in lines:
            line = line.strip()
            if line and len(line) < 100 and not line.startswith('---'):
                return line
                
        return "Untitled Page"
    
    def _extract_text_content(self, parser_output: Dict[str, Any]) -> str:
        """
        Extract plain text content from parser output, ignoring tables and images.
        
        Args:
            parser_output: Output from PDFParser.extract_all()
            
        Returns:
            Concatenated text content from all pages
        """
        text_content = ""
        
        if "text" in parser_output and "pages" in parser_output["text"]:
            pages = parser_output["text"]["pages"]
            
            for page in pages:
                if "content" in page:
                    # Add page number for reference
                    page_num = page.get("page_number", "unknown")
                    text_content += f"\n\n--- Page {page_num} ---\n\n"
                    text_content += page["content"]
        
        return text_content

    def _call_gemini_api(self, prompt: str, system: str = None) -> Dict[str, Any]:
        """
        Call the Gemini API via OpenRouter with the given prompt.
        
        Args:
            prompt: The user message to send to Gemini
            system: Optional system prompt to guide Gemini's behavior
            
        Returns:
            Gemini API response via OpenRouter
        """
        if not system:
            system = "You are a helpful AI assistant specializing in document analysis and information extraction."
        
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

    def _detect_document_type(self, text_content: str) -> Dict[str, Any]:
        """
        Detect the type of document based on its content.
        
        Args:
            text_content: Text content from the PDF
            
        Returns:
            Dictionary with document type information
        """
        system_prompt = """
        You are an expert document classifier. Your task is to analyze document text and determine its type,
        purpose, and any relevant metadata. Provide your analysis in structured JSON format.
        """
        
        # Add company context if available
        company_context = ""
        if self.company_name:
            company_context = f"This document is related to the company '{self.company_name}'."
        
        prompt = f"""
        Analyze the following document text and determine its type and purpose.
        Consider formats like: legal contract, financial report, scientific paper, user manual, 
        technical specification, meeting minutes, policy document, press release, etc.
        {company_context}
        
        For your response, provide the following in JSON format:
        1. "type": The primary document type
        2. "subtypes": Any more specific categorizations (array)
        3. "purpose": The main purpose/function of this document
        4. "confidence": Your confidence level (0-1)
        5. "keywords": Key identifying terms that helped you classify this document (array)
        
        Here is the document text:
        ```
        {text_content[:4000]}  # Limit length for API
        ```
        
        Respond with ONLY a JSON object containing your analysis. No other text.
        """
        
        response = self._call_gemini_api(prompt, system_prompt)
        result = self._parse_json_from_response(response)
        
        return result
    
    def _extract_topics(self, text_content: str, is_page_level: bool = False) -> Dict[str, Any]:
        """
        Extract main topics and themes from the document or page content.
        
        Args:
            text_content: Text content from the PDF or a single page
            is_page_level: Whether this is page-level analysis (vs document-level)
            
        Returns:
            Dictionary with topic information
        """
        system_prompt = """
        You are an expert in topic modeling and thematic analysis.
        Your task is to identify the main topics, themes, and subject matter in documents.
        Provide your analysis in structured JSON format.
        """
        
        # Add company context if available
        company_context = ""
        if self.company_name:
            company_context = f"This document is related to the company '{self.company_name}'. Focus on topics relevant to this company."
        
        if is_page_level:
            # For page-level analysis, use a more focused prompt with fewer topics
            prompt = f"""
            Analyze the following page text and identify the main topics and themes.
            {company_context}
            
            For your response, provide:
            1. "main_topics": An array of 1-3 main topics covered on this page
            2. "keywords": For each topic, provide 3-5 keywords that best represent it
            
            Here is the page text:
            ```
            {text_content[:4000]}
            ```
            
            Respond with ONLY a JSON object containing your topic analysis. No other text.
            """
        else:
            # For document-level analysis, use the original more comprehensive prompt
            prompt = f"""
            Analyze the following document text and identify the main topics and themes.
            {company_context}
            
            For your response, provide:
            1. "main_topics": An array of 3-5 main topics covered in the document
            2. "topic_distribution": For each main topic, provide a relevance score (0-1) indicating how central it is to the document
            3. "keywords": For each topic, provide 5-7 keywords that best represent it
            4. "topic_relationships": Brief description of how the topics relate to each other
            
            Here is the document text:
            ```
            {text_content[:10000]}
            ```
            
            Respond with ONLY a JSON object containing your topic analysis. No other text.
            """
        
        response = self._call_gemini_api(prompt, system_prompt)
        result = self._parse_json_from_response(response)
        
        return result
        
    def _extract_entities(self, text_content: str, document_type: str = "unknown", is_page_level: bool = False) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract named entities from the document or page content based on document type.
        
        Args:
            text_content: Text content from the PDF or a single page
            document_type: The detected document type to customize entity extraction
            is_page_level: Whether this is page-level analysis (vs document-level)
            
        Returns:
            Dictionary of entity types and their instances
        """
        system_prompt = """
        You are an expert in named entity recognition and information extraction.
        Your task is to identify and extract structured information from documents.
        Provide your extraction in structured JSON format.
        """
        
        # Base entity types to extract for all document types
        entity_types = [
            "People (names of individuals)",
            "Organizations (company names, institutions)",
            "Locations (places, addresses)",
            "Dates (any date references)"
        ]
        
        # Add document-type specific entity types
        if document_type == "financial_report":
            entity_types.extend([
                "Financial figures (monetary amounts, percentages, growth rates)",
                "Fiscal years and quarters",
                "Financial metrics (revenue, profit, EBITDA, etc.)",
                "Products and services",
                "Market segments",
                "Currencies and exchange rates"
            ])
        elif document_type == "legal_contract" or document_type == "policy_document":
            entity_types.extend([
                "Legal references (laws, regulations, case references)",
                "Contract clauses",
                "Legal entities (parties, signatories)",
                "Obligations and rights",
                "Effective dates and durations",
                "Jurisdictions"
            ])
        elif document_type == "scientific_paper":
            entity_types.extend([
                "Scientific concepts and terms",
                "Research methods",
                "Measurements and units",
                "Citations and references",
                "Institutions and funding sources",
                "Scientific equipment and materials"
            ])
        elif document_type == "technical_document" or document_type == "user_manual":
            entity_types.extend([
                "Technical specifications",
                "Product components",
                "Software versions",
                "Technical parameters",
                "Error codes",
                "Technical procedures"
            ])
        else:
            # Generic additional entity types for unknown document types
            entity_types.extend([
                "Financial figures (monetary amounts, percentages)",
                "Products or services",
                "Legal references (laws, regulations, case references)"
            ])
        
        # Create the prompt with the selected entity types
        company_context = ""
        if self.company_name:
            company_context = f"This document is related to the company '{self.company_name}'. Pay special attention to mentions of this company and its subsidiaries, products, executives, and related entities."
        
        if is_page_level:
            # For page-level analysis, use a more focused prompt
            prompt = f"""
            Extract named entities from the following page text.
            This is from a document that appears to be a {document_type} document.
            {company_context}
            
            Include the following entity types:
            {chr(10).join(f"- {entity_type}" for entity_type in entity_types)}
            
            For each entity, provide:
            1. "text": The exact entity text
            2. "category": The entity type
            3. "confidence": Your confidence level (0-1)
            
            Here is the page text:
            ```
            {text_content[:4000]}
            ```
            
            Respond with ONLY a JSON object containing the extracted entities, grouped by category. No other text.
            """
        else:
            # For document-level analysis, use the original more comprehensive prompt
            prompt = f"""
            Extract named entities from the following document text.
            This appears to be a {document_type} document.
            {company_context}
            
            Include the following entity types:
            {chr(10).join(f"- {entity_type}" for entity_type in entity_types)}
            
            For each entity, provide:
            1. "text": The exact entity text
            2. "category": The entity type
            3. "context": Brief surrounding context
            4. "page": Page number if mentioned (otherwise "unknown")
            5. "confidence": Your confidence level (0-1)
            
            Here is the document text:
            ```
            {text_content[:8000]}
            ```
            
            Respond with ONLY a JSON object containing the extracted entities, grouped by category. No other text.
            """
        
        response = self._call_gemini_api(prompt, system_prompt)
        result = self._parse_json_from_response(response)
        
        # Ensure we have a properly structured result
        if not isinstance(result, dict):
            return {"error": "Failed to extract entities properly"}
        
        return result
    
    def _extract_sections(self, text_content: str) -> List[Dict[str, Any]]:
        """
        Extract the document's section structure.
        
        Args:
            text_content: Text content from the PDF
            
        Returns:
            List of document sections with hierarchical structure
        """
        system_prompt = """
        You are an expert in document structure analysis.
        Your task is to identify the hierarchical structure of documents, including sections, subsections, and their content.
        Provide your analysis in structured JSON format.
        """
        
        # Add company context if available
        company_context = ""
        if self.company_name:
            company_context = f"This document is related to the company '{self.company_name}'. Pay special attention to sections that discuss this company's operations, financials, or strategic initiatives."
        
        prompt = f"""
        Analyze the following document text and extract its hierarchical structure.
        Identify sections, subsections, and their relationships.
        {company_context}
        
        For each section, provide:
        1. "title": The section title/heading
        2. "level": The heading level (1 for main sections, 2 for subsections, etc.)
        3. "content_summary": A brief summary of the section's content (1-2 sentences)
        4. "page_range": The page numbers this section spans (if identifiable)
        5. "subsections": An array of subsections (with the same structure)
        
        Here is the document text:
        ```
        {text_content[:8000]}  # Using more text for section extraction
        ```
        
        Respond with ONLY a JSON array containing the sections structure. No other text.
        """
        
        response = self._call_gemini_api(prompt, system_prompt)
        result = self._parse_json_from_response(response)
        
        # Ensure we have a properly structured result
        if isinstance(result, dict) and "error" not in result:
            # If the result is a dict with an array field, extract it
            for key in result:
                if isinstance(result[key], list):
                    return result[key]
            
        # If it's already a list, use it directly
        if isinstance(result, list):
            return result
            
        # Otherwise, return an empty structure
        return []
    
    def _generate_summary(self, text_content: str) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of the document.
        
        Args:
            text_content: Text content from the PDF
            
        Returns:
            Dictionary with different summary types
        """
        system_prompt = """
        You are an expert in document summarization.
        Your task is to create concise, accurate summaries of documents at different levels of detail.
        Provide your summaries in structured JSON format.
        """
        
        # Add company context if available
        company_context = ""
        if self.company_name:
            company_context = f"This document is related to the company '{self.company_name}'. Focus on information relevant to this company in your summary."
        
        prompt = f"""
        Generate a comprehensive summary of the following document text.
        {company_context}
        
        Provide the following in your response:
        1. "brief_summary": A 1-2 sentence summary of the entire document
        2. "executive_summary": A paragraph (3-5 sentences) highlighting key points
        3. "detailed_summary": A more comprehensive summary (up to 500 words)
        4. "key_points": An array of the 5-7 most important points from the document
        5. "audience": Who this document appears to be intended for
        
        Here is the document text:
        ```
        {text_content[:12000]}  # Using more text for summary generation
        ```
        
        Respond with ONLY a JSON object containing the summaries. No other text.
        """
        
        response = self._call_gemini_api(prompt, system_prompt)
        result = self._parse_json_from_response(response)
        
        return result
    
    def _extract_key_information(self, text_content: str, document_type: str) -> Dict[str, Any]:
        """
        Extract key information based on the document type.
        
        Args:
            text_content: Text content from the PDF
            document_type: The detected document type
            
        Returns:
            Dictionary with key information specific to the document type
        """
        # Base system prompt
        system_prompt = """
        You are an expert in information extraction from specialized documents.
        Your task is to identify and extract key information based on the document type.
        Provide your extraction in structured JSON format.
        """
        
        # Add company context if available
        company_context = ""
        if self.company_name:
            company_context = f"This document is related to the company '{self.company_name}'. Focus on information specific to this company."
        
        # Customize the prompt based on document type
        if document_type == "financial_report":
            prompt = f"""
            Extract key financial information from this financial report:
            - Financial highlights (revenue, profit, growth rates)
            - Time period covered
            - Company name and identifiers
            - Key financial ratios
            - Significant events or changes mentioned
            
            {company_context}
            
            Here is the document text:
            ```
            {text_content[:10000]}
            ```
            
            Respond with ONLY a JSON object containing the extracted financial information. No other text.
            """
            
        elif document_type == "legal_contract" or document_type == "policy_document":
            prompt = f"""
            Extract key legal information from this document:
            - Parties involved
            - Effective date and duration
            - Key obligations and rights
            - Important clauses or conditions
            - Termination conditions
            - Governing law
            
            Here is the document text:
            ```
            {text_content[:10000]}
            ```
            
            Respond with ONLY a JSON object containing the extracted legal information. No other text.
            """
            
        elif document_type == "scientific_paper":
            prompt = f"""
            Extract key scientific information from this paper:
            - Research question or hypothesis
            - Methodology
            - Key findings
            - Conclusions
            - Authors and institutions
            - Publication information
            
            Here is the document text:
            ```
            {text_content[:10000]}
            ```
            
            Respond with ONLY a JSON object containing the extracted scientific information. No other text.
            """
            
        else:
            # Generic extraction for other document types
            prompt = f"""
            Extract key information from this document of type "{document_type}":
            - Main topic or purpose
            - Key entities mentioned (people, organizations)
            - Important dates
            - Critical facts or figures
            - Action items or next steps
            - Document metadata (author, date, version if available)
            
            Here is the document text:
            ```
            {text_content[:10000]}
            ```
            
            Respond with ONLY a JSON object containing the extracted key information. No other text.
            """
        
        response = self._call_gemini_api(prompt, system_prompt)
        result = self._parse_json_from_response(response)
        
        return result
    
    def _run_custom_analysis(self, text_content: str, analysis_name: str, custom_prompt: str) -> Dict[str, Any]:
        """
        Run a custom analysis specified in the configuration.
        
        Args:
            text_content: Text content from the PDF
            analysis_name: Name of the custom analysis
            custom_prompt: Custom prompt template
            
        Returns:
            Result of the custom analysis
        """
        logger.info(f"Running custom analysis: {analysis_name}")
        
        try:
            # Prepare the prompt by inserting the text content
            # Try with document_text first (as used in the config)
            formatted_prompt = custom_prompt.format(document_text=text_content[:10000])
            logger.info(f"Successfully formatted custom prompt using 'document_text' placeholder")
        except KeyError:
            try:
                # Fall back to text_content if document_text fails
                formatted_prompt = custom_prompt.format(text_content=text_content[:10000])
                logger.info(f"Successfully formatted custom prompt using 'text_content' placeholder")
            except KeyError as e:
                logger.error(f"Error formatting custom prompt for {analysis_name}: {str(e)}")
                logger.error(f"Make sure your custom prompt uses {{document_text}} as the placeholder")
                return {"error": f"Failed to format custom prompt: {str(e)}"}
        
        system_prompt = f"You are an expert in {analysis_name}. Provide your analysis in structured JSON format."
        
        try:
            logger.info(f"Calling Gemini API for custom analysis: {analysis_name}")
            response = self._call_gemini_api(formatted_prompt, system_prompt)
            logger.info(f"Received response from Gemini API for {analysis_name}")
            
            logger.info(f"Parsing JSON from response for {analysis_name}")
            result = self._parse_json_from_response(response)
            logger.info(f"Successfully parsed JSON for {analysis_name}")
            
            return result
        except Exception as e:
            logger.error(f"Error in custom analysis {analysis_name}: {str(e)}")
            return {"error": f"Failed to complete custom analysis: {str(e)}"}
    
    def _save_enhanced_output(self, enhanced_output: Dict[str, Any], output_path: str) -> None:
        """
        Save the enhanced output to a JSON file.
        
        Args:
            enhanced_output: The enhanced PDF content
            output_path: Path to save the output
        """
        try:
            # Ensure the output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(enhanced_output, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Failed to save enhanced output: {str(e)}")
