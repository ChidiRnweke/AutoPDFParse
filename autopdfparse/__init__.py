"""
AutoPDFParse - A package for extracting content from PDF documents using AI.
"""

from .exceptions import (
    APIError,
    AutoPDFParseError,
    FileAccessError,
    ModelError,
    PDFParsingError,
)
from .models import ParsedData, ParsedPDFResult, PDFPage, VisualModelDecision
from .parsers import OpenAIParser, PDFParser
from .services import OpenAIVisionService, VisionService

__version__ = "0.1.0"

__all__ = [
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
    # Services
    "VisionService",
    "OpenAIVisionService",
]
