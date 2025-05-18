"""
AutoPDFParse - A package for extracting content from PDF documents using AI.
"""

from .config import Config
from .exceptions import (
    APIError,
    AutoPDFParseError,
    FileAccessError,
    ModelError,
    PDFParsingError,
)
from .models import ParsedData, ParsedPDFResult, PDFPage, VisualModelDecision
from .providers import AnthropicParser, GeminiParser, OpenAIParser
from .services import PDFParser, VisionService
from .sync import services

__version__ = "0.1.0"

__all__ = [
    # Config
    "Config",
    # Exceptions
    "AutoPDFParseError",
    "PDFParsingError",
    "APIError",
    "ModelError",
    "FileAccessError",
    # Models
    "PDFPage",
    "ParsedPDFResult",
    "ParsedData",
    "VisualModelDecision",
    # Parsers
    "PDFParser",
    "OpenAIParser",
    "AnthropicParser",
    "GeminiParser",
    # Services
    "VisionService",
    # Sync API
    "services",
]
