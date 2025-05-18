"""
Exceptions for the AutoPDFParse package.
"""


class AutoPDFParseError(Exception):
    """Base exception for all AutoPDFParse errors."""

    pass


class PDFParsingError(AutoPDFParseError):
    """Error occurred during PDF parsing."""

    pass


class APIError(AutoPDFParseError):
    """Error occurred during API calls."""

    pass


class ModelError(AutoPDFParseError):
    """Error related to model selection or availability."""

    pass


class FileAccessError(AutoPDFParseError):
    """Error accessing or reading file."""

    pass
