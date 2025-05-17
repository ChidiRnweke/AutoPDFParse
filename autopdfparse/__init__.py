"""
AutoPDFParse - A package for parsing PDF documents using OpenAI models.
"""

from .exceptions import (
    APIError,
    AutoPDFParseError,
    FileAccessError,
    ModelError,
    PDFParsingError,
)
from .parse import (
    OpenAIParser,
    ParsedData,
    ParsedPDFResult,
    PDFPage,
    PDFParser,
    VisualModelDecision,
)

__all__ = [
    "OpenAIParser",
    "ParsedData",
    "ParsedPDFResult",
    "PDFPage",
    "PDFParser",
    "VisualModelDecision",
    "APIError",
    "AutoPDFParseError",
    "FileAccessError",
    "ModelError",
    "PDFParsingError",
]

__version__ = "0.1.0"
