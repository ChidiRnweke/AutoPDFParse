"""
Base parser implementations.
"""

from typing import Protocol

from ..models import ParsedPDFResult


class PDFParser(Protocol):
    """Protocol defining the interface for PDF parsers."""

    async def parse(self) -> ParsedPDFResult: ...
