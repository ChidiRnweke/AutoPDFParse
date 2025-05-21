"""
Base parser implementations.
"""

import asyncio
import base64
import logging
from dataclasses import dataclass

import pymupdf

from ..exceptions import PDFParsingError
from ..models import ParsedData, ParsedPDFResult, PDFPage
from .vision import VisionService


@dataclass
class PDFParser:
    """
    Base implementation of a PDF parser that uses a vision service.

    This parser handles the core PDF parsing logic but delegates visual
    analysis to a pluggable vision service.
    """

    vision_service: VisionService

    async def parse_file(self, file_path: str) -> ParsedPDFResult:
        """
        Parse the PDF document using the provided vision service.

        Returns:
            ParsedPDFResult with extracted content

        Raises:
            PDFParsingError: If parsing fails
        """
        try:
            with open(file_path, "rb") as f:
                pdf_content = f.read()
        except FileNotFoundError:
            raise PDFParsingError(f"File not found: {file_path}")
        except Exception as e:
            raise PDFParsingError(f"Failed to read file: {file_path}. Error: {str(e)}")
        return await self.parse_bytes(pdf_content)

    async def parse_bytes(self, pdf_content: bytes) -> ParsedPDFResult:
        """
        Parse the PDF document using the provided vision service.

        Returns:
            ParsedPDFResult with extracted content

        Raises:
            PDFParsingError: If parsing fails
        """
        try:
            document = pymupdf.open(stream=pdf_content)
            images: list[pymupdf.Pixmap] = [page.get_pixmap() for page in document]  # type: ignore
            page_texts: list[str] = [page.get_text() for page in document]  # type: ignore

            if not images:
                return ParsedPDFResult(pages=[])

            async with asyncio.TaskGroup() as tg:
                tasks: list[asyncio.Task[ParsedData]] = []
                for i, (text, image) in enumerate(zip(page_texts, images)):
                    tasks.append(tg.create_task(self._parse_page(text, image, i + 1)))
            results = [task.result() for task in tasks]
            pages = [
                PDFPage(
                    content=result.content,
                    page_number=i + 1,
                    _from_llm=result._from_llm,
                )
                for i, result in enumerate(results)
            ]
            return ParsedPDFResult(pages=pages)
        except pymupdf.FileDataError as e:
            raise PDFParsingError(f"Invalid PDF format: {str(e)}")
        except Exception as e:
            raise PDFParsingError(f"Failed to parse PDF: {str(e)}")

    async def _parse_page(
        self, page_text: str, page_as_image: pymupdf.Pixmap, page_num: int
    ) -> ParsedData:
        try:
            # Convert image to base64 string
            image_bytes = page_as_image.pil_tobytes("png")
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")

            is_layout_dependent = await self.vision_service.is_layout_dependent(
                image_base64
            )
            if is_layout_dependent:
                content = await self.vision_service.describe_image_content(image_base64)
                from_llm = True
            else:
                content = page_text
                from_llm = False
            return ParsedData(content=content, _from_llm=from_llm)
        except Exception as e:
            logging.error(f"Error parsing page {page_num}: {str(e)}")
            raise PDFParsingError(f"Error processing page {page_num}. ")
