"""
OpenAI-based PDF parser implementation.
"""

from typing import Optional

from ..services.vision import OpenAIVisionService
from .base import PDFParser


class OpenAIParser:
    """
    Factory class for creating PDF parsers that use OpenAI's vision models.

    This class provides convenience methods for creating BasePDFParser instances
    that are configured to use OpenAI's vision services.
    """

    @classmethod
    async def from_file(
        cls,
        file_path: str,
        api_key: str,
        description_model: Optional[str] = None,
        visual_model: Optional[str] = None,
        retries: int = 3,
    ) -> PDFParser:
        """
        Create a PDF parser from a file path using OpenAI vision services.

        Args:
            file_path: Path to the PDF file
            api_key: OpenAI API key
            description_model: Model to use for describing content
            visual_model: Model to use for layout dependency detection
            retries: Number of retries for API calls

        Returns:
            BasePDFParser instance configured with OpenAIVisionService
        """
        vision_service = OpenAIVisionService.create(
            api_key=api_key,
            description_model=description_model,
            visual_model=visual_model,
            retries=retries,
        )

        return await PDFParser.create(
            file_path=file_path, vision_service=vision_service, image_retries=retries
        )

    @classmethod
    async def from_bytes(
        cls,
        pdf_content: bytes,
        api_key: str,
        description_model: Optional[str] = None,
        visual_model: Optional[str] = None,
        retries: int = 3,
    ) -> PDFParser:
        """
        Create a PDF parser from bytes using OpenAI vision services.

        Args:
            pdf_content: PDF content as bytes
            api_key: OpenAI API key
            description_model: Model to use for describing content
            visual_model: Model to use for layout dependency detection
            retries: Number of retries for API calls

        Returns:
            BasePDFParser instance configured with OpenAIVisionService
        """
        vision_service = OpenAIVisionService.create(
            api_key=api_key,
            description_model=description_model,
            visual_model=visual_model,
            retries=retries,
        )

        return await PDFParser.from_bytes(
            pdf_content=pdf_content,
            vision_service=vision_service,
            image_retries=retries,
        )
