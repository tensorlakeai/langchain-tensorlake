import time
import os
from dotenv import load_dotenv
from typing import Optional, Type, Union
import asyncio
from langchain_core.tools import StructuredTool
from pydantic import Field, BaseModel

from tensorlake.documentai import DocumentAI, ParsingOptions
from tensorlake.documentai.parse import (
    ChunkingStrategy,
    TableParsingStrategy,
    TableOutputMode,
    ExtractionOptions,
    ModelProvider,
    FormDetectionMode
)

load_dotenv()

TENSORLAKE_API_KEY = os.getenv("TENSORLAKE_API_KEY")


class DocumentParserOptions(BaseModel):
    """
    Comprehensive options for parsing documents with Tensorlake AI.
    Choose parameters based on the user's question and document analysis needs.
    """

    # Chunking options
    chunking_strategy: Optional[ChunkingStrategy] = Field(
        default=ChunkingStrategy.PAGE,
        description="""Strategy for breaking down the document:
        - NONE: Keep document as one piece (good for short docs or when you need full context)
        - PAGE: Split by pages (default, good for most documents)
        - SECTION_HEADER: Split by headers (good for structured documents like reports)"""
    )

    # Table parsing options
    table_parsing_strategy: TableParsingStrategy = Field(
        default=TableParsingStrategy.VLM,
        description="""Algorithm for parsing tables:
        - TSR: Use for clean, well-formatted tables with clear borders
        - VLM: Use for complex tables, handwritten tables, or tables in images (more accurate but slower)
        Choose VLM for financial reports, forms, or complex layouts."""
    )

    table_output_mode: TableOutputMode = Field(
        default=TableOutputMode.MARKDOWN,
        description="""Format for table output:
        - JSON: For structured data extraction and analysis
        - MARKDOWN: For readable text format (good for summaries)
        - HTML: For preserving complex formatting"""
    )

    table_parsing_prompt: Optional[str] = Field(
        default=None,
        description="""Custom prompt to guide table parsing. Examples:
        - "Focus on financial data and numbers"
        - "Extract all form fields and their values"
        - "Parse this as a data table with headers" 
        Use when you need specific focus on certain table elements."""
    )

    table_summary: bool = Field(
        default=False,
        description="""Generate summaries of tables. Set to True when:
        - User asks for table analysis or insights
        - Document has many tables that need summarization
        - User wants to understand table contents without raw data"""
    )

    # Figure and image options
    figure_summary: bool = Field(
        default=False,
        description="""Generate summaries of figures, charts, and images. Set to True when:
        - User asks about charts, graphs, or visual elements
        - Document contains important diagrams or infographics
        - User needs to understand visual data representations"""
    )

    figure_summarization_prompt: Optional[str] = Field(
        default=None,
        description="""Custom prompt for figure analysis. Examples:
        - "Describe the trends shown in these charts"
        - "Extract data points from graphs"
        - "Identify key visual elements and their meaning"
        Use for specific figure analysis requirements."""
    )

    # Page and layout options
    page_range: Optional[str] = Field(
        default=None,
        description="""Specific pages to parse (e.g., '1-5', '1,3,5', '10-end').
        Use when user asks about specific sections or pages of the document."""
    )

    skew_correction: bool = Field(
        default=False,
        description="""Apply skew correction to scanned documents.
        Set to True for poorly scanned documents or photos of documents."""
    )

    disable_layout_detection: bool = Field(
        default=False,
        description="""Disable automatic layout detection.
        Set to True only if layout detection is causing issues with document structure."""
    )

    # Signature and form detection
    detect_signature: bool = Field(
        default=False,
        description="""Detect signatures in the document. Set to True when:
        - User asks about signatures, signing, or authentication
        - Document is a contract, agreement, or legal document
        - User needs to verify document execution"""
    )

    form_detection_mode: FormDetectionMode = Field(
        default=FormDetectionMode.OBJECT_DETECTION,
        description="""Algorithm for form detection:
        - OBJECT_DETECTION: Fast, good for standard forms
        - VLM: More accurate for complex or handwritten forms
        Use VLM for handwritten forms or complex layouts."""
    )

    # Structured extraction options
    extraction_schema: Optional[Union[Type[BaseModel], dict, str]] = Field(
        default=None,
        description="""JSON schema for structured data extraction. Use when:
        - User wants specific data fields extracted
        - Need to convert unstructured document to structured data
        - Building data pipelines or databases from documents
        Example: {"name": "string", "date": "date", "amount": "number"}"""
    )

    extraction_prompt: Optional[str] = Field(
        default=None,
        description="""Custom prompt for structured extraction. Examples:
        - "Extract all personal information and contact details"
        - "Find all financial transactions and amounts"
        - "Get all dates, names, and locations mentioned"
        Use to guide what specific information to extract."""
    )

    extraction_model_provider: ModelProvider = Field(
        default=ModelProvider.TENSORLAKE,
        description="""Model for extraction:
        - TENSORLAKE: Fast, cost-effective (default)
        - SONNET: High accuracy for complex extraction
        - GPT4OMINI: Good balance of speed and accuracy"""
    )

    skip_ocr: bool = Field(
        default=False,
        description="""Skip OCR for text-based PDFs.
        Set to True for digital PDFs to speed up processing and preserve text quality."""
    )

    timeout_seconds: int = Field(
        default=300,
        description="Maximum processing time in seconds (increase for large documents)."
    )


def document_to_markdown_converter(
        path: str,
        chunking_strategy: Optional[ChunkingStrategy] = ChunkingStrategy.PAGE,
        table_parsing_strategy: TableParsingStrategy = TableParsingStrategy.VLM,
        table_output_mode: TableOutputMode = TableOutputMode.MARKDOWN,
        table_parsing_prompt: Optional[str] = None,
        table_summary: bool = False,
        figure_summary: bool = False,
        figure_summarization_prompt: Optional[str] = None,
        page_range: Optional[str] = None,
        skew_correction: bool = False,
        disable_layout_detection: bool = False,
        detect_signature: bool = False,
        form_detection_mode: FormDetectionMode = FormDetectionMode.OBJECT_DETECTION,
        extraction_schema: Optional[Union[Type[BaseModel], dict, str]] = None,
        extraction_prompt: Optional[str] = None,
        extraction_model_provider: ModelProvider = ModelProvider.TENSORLAKE,
        skip_ocr: bool = False,
        timeout_seconds: int = 300
) -> str:
    """
    Convert a document to markdown using Tensorlake's DocumentAI.

    Args:
        path: Path to the document file to parse (supports PDF, DOCX, images, etc.)
        All parameters are individual fields that the agent can set based on the question.

    Returns:
        str: The parsed document in markdown format, or error message if failed

    Raises:
        ValueError: If API key is not configured
        Exception: If document processing fails
    """
    if not TENSORLAKE_API_KEY:
        return "Error: TENSORLAKE_API_KEY environment variable is not set"

    try:
        # Initialize DocumentAI client
        doc_ai = DocumentAI(api_key=TENSORLAKE_API_KEY)

        # Upload document to TensorLake
        file_id = doc_ai.upload(path=path)

        # Configure parsing options based on user input
        parsing_options = ParsingOptions(
            chunking_strategy=chunking_strategy,
            table_parsing_strategy=table_parsing_strategy,
            table_output_mode=table_output_mode,
            table_parsing_prompt=table_parsing_prompt,
            table_summary=table_summary,
            figure_summary=figure_summary,
            figure_summarization_prompt=figure_summarization_prompt,
            page_range=page_range,
            skew_correction=skew_correction,
            disable_layout_detection=disable_layout_detection,
            detect_signature=detect_signature,
            form_detection_mode=form_detection_mode,
            deliver_webhook=deliver_webhook
        )

        # Add extraction options if schema is provided
        schema = extraction_schema
        if isinstance(schema, dict):
            import json
            schema = json.dumps(schema)
        if schema:
            parsing_options.extraction_options = ExtractionOptions(
                schema=schema,
                prompt=extraction_prompt,
                provider=extraction_model_provider,
                skip_ocr=skip_ocr
            )
        elif skip_ocr:
            parsing_options.extraction_options = ExtractionOptions(
                provider=extraction_model_provider,
                skip_ocr=skip_ocr
            )

        # Start parsing job
        job_id = doc_ai.parse(file_id, options=parsing_options)

        # Poll for completion with configurable timeout
        start_time = time.time()
        max_wait_time = timeout_seconds

        while time.time() - start_time < max_wait_time:
            result = doc_ai.get_job(job_id)

            if result.status in ["pending", "processing"]:
                time.sleep(5)  # Wait 5 seconds before checking again
            elif result.status == "successful":
                # Return the parsed content
                if hasattr(result, 'content') and result.content:
                    return result.content
                elif hasattr(result, 'markdown') and result.markdown:
                    return result.markdown
                else:
                    return str(result)  # Fallback to string representation
            else:
                return f"Document parsing failed with status: {result.status}"

        # Timeout reached
        return f"Document processing timeout after {max_wait_time} seconds. Job ID: {job_id}"

    except Exception as e:
        return f"Error processing document: {str(e)}"


async def document_to_markdown_converter_async(path: str, **kwargs) -> str:
    """Asynchronous version of document to markdown converter."""
    return await asyncio.to_thread(document_to_markdown_converter, path, **kwargs)


# Create the Document to Markdown tool using StructuredTool
document_markdown_tool = StructuredTool.from_function(
    func=document_to_markdown_converter,
    coroutine=document_to_markdown_converter_async,
    name="DocumentToMarkdownConverter",
    description="""Advanced document parser that converts documents to markdown with AI-powered analysis.

**CAPABILITIES:**
- Parse PDFs, DOCX, images, and other document formats
- Extract and analyze tables with different strategies (TSR for clean tables, VLM for complex ones)
- Summarize figures, charts, and visual elements
- Detect signatures and form fields
- Extract structured data using custom schemas
- Handle scanned documents with OCR and skew correction

**WHEN TO USE DIFFERENT PARAMETERS:**
- Questions about "signatures" or "signed documents" → set detect_signature=True
- Questions about "tables", "data", or "financial information" → use table parsing with summaries
- Questions about "charts", "graphs", or "visual elements" → set figure_summary=True
- Questions about "forms" or "form fields" → use form detection
- Questions asking for "structured data" or "specific fields" → use extraction_schema
- Questions about "specific pages" → set page_range
- Scanned or poor quality documents → set skew_correction=True

Choose parameters intelligently based on what the user is asking about.""",
    args_schema=DocumentParserOptions,
    return_direct=False,
    handle_tool_error="Document parsing failed. Please verify the file path and your Tensorlake API key."
)
