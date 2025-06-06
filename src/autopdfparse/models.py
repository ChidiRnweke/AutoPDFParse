"""
Data models for the AutoPDFParse package.
"""

from dataclasses import dataclass
from typing import List

from pydantic import BaseModel


@dataclass
class PDFPage:
    """
    Represents a single page of a parsed PDF document.

    Attributes:
        content: The extracted text content of the page
        page_number: The page number (1-indexed)
        _from_llm: Whether the content was generated by an LLM
    """

    content: str
    page_number: int
    _from_llm: bool


@dataclass
class ParsedPDFResult:
    """
    Container for all parsed pages of a PDF document.

    Attributes:
        pages: List of parsed PDF pages
    """

    pages: list[PDFPage]

    def get_all_content(self) -> str:
        """Returns all page content concatenated into a single string."""
        return "\n\n".join(page.content for page in self.pages)


@dataclass
class ParsedData:
    """
    Internal data structure for parsed page content.

    Attributes:
        content: The extracted text content
        _from_llm: Whether the content was generated by an LLM
    """

    content: str
    _from_llm: bool


class VisualModelDecision(BaseModel):
    """
    Pydantic model for the layout dependency decision.

    Attributes:
        content_is_layout_dependent: Whether the content depends on its layout
    """

    content_is_layout_dependent: bool
