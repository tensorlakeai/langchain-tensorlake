import os
import time
from typing import Optional

from dotenv import load_dotenv
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from tensorlake.documentai import DocumentAI
from tensorlake.documentai.models import (
    EnrichmentOptions, 
    ParsingOptions,
    ParseStatus
)
from tensorlake.documentai.models.enums import (
    ChunkingStrategy,
    ParseStatus,
    TableOutputMode,
    TableParsingFormat,
)

load_dotenv()

TENSORLAKE_API_KEY = os.getenv("TENSORLAKE_API_KEY")

class DocumentParserOptions(BaseModel):
    """Comprehensive options for parsing a document with Tensorlake."""

    # Chunking options
    chunking_strategy: Optional[ChunkingStrategy] = Field(
        default=ChunkingStrategy.PAGE,
        description="Strategy for chunking the document (NONE, PAGE, or SECTION_HEADER)"
    )

    # Table parsing options
    table_parsing_format: TableParsingFormat = Field(
        default=TableParsingFormat.VLM,
        description="Algorithm for parsing tables (TSR for structured tables, VLM for complex/unstructured tables)"
    )
    table_output_mode: TableOutputMode = Field(
        default=TableOutputMode.MARKDOWN,
        description="Format for table output (JSON, MARKDOWN, or HTML)"
    )
    table_summarization_prompt: Optional[str] = Field(
        default=None,
        description="Custom prompt to guide table parsing"
    )
    table_summarization: bool = Field(
        default=False,
        description="Whether to generate summaries of tables"
    )

    # Figure and image options
    figure_summarization: bool = Field(
        default=False,
        description="Whether to generate summaries of figures and images"
    )
    figure_summarization_prompt: Optional[str] = Field(
        default=None,
        description="Custom prompt for figure summarization"
    )

    # Page and layout options
    page_range: Optional[str] = Field(
        default=None,
        description="Specific page range to parse (e.g., '1-5' or '1,3,5')"
    )
    skew_detection: bool = Field(
        default=False,
        description="Whether to apply skew correction to scanned documents"
    )
    disable_layout_detection: bool = Field(
        default=False,
        description="Whether to disable automatic layout detection"
    )

    # Signature detection
    signature_detection: bool = Field(
        default=False,
        description="Whether to detect the presence of signatures in the document"
    )

    # Remove strikethrough
    remove_strikethrough_lines: bool = Field(
        default=False,
        description="Whether to remove strikethrough text from the document"
    )

    # Processing timeout
    timeout_seconds: int = Field(
        default=300,
        description="Maximum time to wait for processing completion (in seconds)"
    )


def document_to_markdown_converter(path: str, options: DocumentParserOptions) -> str:
    """
    Convert a document to markdown using Tensorlake's DocumentAI.

    Args:
        path: Path to the document file to parse (supports PDF, DOCX, images, etc.) or HTTP, HTTPS URL
        options: DocumentParserOptions object containing all parsing configuration

    Returns:
        str: The parsed document in markdown format, or error message if failed

    Raises:
        ValueError: If API key is not configured
        Exception: If document processing fails
    """

    debug = False

    # Ensure the TENSORLAKE_API_KEY is set
    TENSORLAKE_API_KEY = os.getenv("TENSORLAKE_API_KEY")

    if not TENSORLAKE_API_KEY:
        return "Error: TENSORLAKE_API_KEY environment variable is not set"

    try:
        # Initialize DocumentAI client
        doc_ai = DocumentAI(api_key=TENSORLAKE_API_KEY)

        # Upload document to TensorLake
        if path.startswith("http") or path.startswith("https"):
            data = path
        elif os.path.isfile(path):
            data = doc_ai.upload(path=path)
        else:
            data = path

        parsing_options=ParsingOptions(
            remove_strikethrough_lines=options.remove_strikethrough_lines,
            signature_detection=options.signature_detection,
            skew_detection=options.skew_detection,
            table_output_mode=options.table_output_mode,
            table_parsing_format=options.table_parsing_format,
            disable_layout_detection=options.disable_layout_detection,
            chunking_strategy=options.chunking_strategy,
            page_range=options.page_range,
        )

        if debug: print("initialized parsing_options:", parsing_options)
        
        enrichment_options=EnrichmentOptions(
            figure_summarization=options.figure_summarization,
            figure_summarization_prompt=options.figure_summarization_prompt,
            table_summarization=options.table_summarization,
            table_summarization_prompt=options.table_summarization_prompt,
        )

        if debug: print("initialized enrichment_options:", enrichment_options)
        
        # Start parsing job
        parse_id = doc_ai.parse(file=data, parsing_options=parsing_options, enrichment_options=enrichment_options)
        if debug: print("Started parsing job with ID:", parse_id)

        # Poll for completion with configurable timeout
        start_time = time.time()
        max_wait_time = options.timeout_seconds

        while time.time() - start_time < max_wait_time:
            result = doc_ai.get_parsed_result(parse_id)
            if debug: print('Current result:', result)
            if debug: print("result.status:", result.status)

            if result.status in [ParseStatus.PENDING, ParseStatus.PROCESSING]:
                time.sleep(5)  # Wait 5 seconds before checking again
            elif result.status == ParseStatus.SUCCESSFUL:
                # Return the parsed document
                if 'chunks' in result and result.chunks:
                    if debug: print("returning result.markdown")
                    return result.chunks
                else:
                    if debug: print("returning result as string")
                    return str(result)  # Fallback to string representation
            else:
                if debug: print("Returning because document parsing failed with status:", result.status)
                return f"Document parsing failed with status: {result.status}"

        # Timeout reached
        if debug: print("Returning timeout message")
        return f"Document processing timeout after {max_wait_time} seconds. Job ID: {parse_id}"

    except Exception as e:
        if debug: print("Returning error message due to exception:", str(e))
        return f"Error processing document: {str(e)}"


async def document_to_markdown_converter_async(path: str, options: DocumentParserOptions) -> str:
    """Asynchronous version of document to markdown converter."""
    import asyncio
    return await asyncio.to_thread(document_to_markdown_converter, path, options)


# Create the Document to Markdown tool using StructuredTool
document_markdown_tool = StructuredTool.from_function(
    func=document_to_markdown_converter,
    coroutine=document_to_markdown_converter_async,
    name="DocumentToMarkdownConverter",
    description="Convert documents (PDF, DOCX, images, etc.) to markdown using Tensorlake AI. Supports tables, figures, signatures, and structured extraction.",
    return_direct=False,
    handle_tool_error="Document parsing failed. Please verify the file path and your Tensorlake API key."
)
