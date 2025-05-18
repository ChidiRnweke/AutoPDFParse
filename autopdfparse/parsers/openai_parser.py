"""
OpenAI-based PDF parser implementation.
"""

import asyncio
import base64
import logging
from dataclasses import dataclass
from typing import ClassVar, Optional

import pymupdf
from openai import AsyncOpenAI

from ..exceptions import APIError, PDFParsingError
from ..models import (
    ParsedData,
    ParsedPDFResult,
    PDFPage,
    PDFParser,
    VisualModelDecision,
)


@dataclass
class OpenAIParser(PDFParser):
    """
    A PDF parser that uses OpenAI's vision models to extract content from PDFs.

    The parser automatically detects if a page contains layout-dependent content
    (like tables, charts, complex formatting) and uses vision models in those cases.
    """

    api_key: str
    pdf_content: bytes
    description_model: str
    visual_model: str
    image_retries: int = 3

    # Default models to use
    DEFAULT_DESCRIPTION_MODEL: ClassVar[str] = "gpt-4.1"
    DEFAULT_VISUAL_MODEL: ClassVar[str] = "gpt-4.1-mini"

    @classmethod
    def from_file(
        cls,
        file_path: str,
        api_key: str,
        description_model: Optional[str] = None,
        visual_model: Optional[str] = None,
        image_retries: int = 3,
    ) -> "OpenAIParser":
        """
        Creates an OpenAIParser instance from a file path.

        Args:
            file_path: Path to the PDF file
            api_key: OpenAI API key
            description_model: Model to use for describing content
            visual_model: Model to use for layout dependency detection
            image_retries: Number of retries for image processing

        Returns:
            OpenAIParser instance

        Raises:
            PDFParsingError: If the file cannot be read or is not a valid PDF
        """
        try:
            with open(file_path, "rb") as f:
                pdf_content = f.read()

            return cls(
                api_key=api_key,
                pdf_content=pdf_content,
                description_model=description_model or cls.DEFAULT_DESCRIPTION_MODEL,
                visual_model=visual_model or cls.DEFAULT_VISUAL_MODEL,
                image_retries=image_retries,
            )
        except FileNotFoundError:
            raise PDFParsingError(f"File not found: {file_path}")
        except Exception as e:
            raise PDFParsingError(f"Error reading PDF file: {str(e)}")

    @classmethod
    def from_bytes(
        cls,
        pdf_content: bytes,
        api_key: str,
        description_model: Optional[str] = None,
        visual_model: Optional[str] = None,
        image_retries: int = 3,
    ) -> "OpenAIParser":
        """
        Creates an OpenAIParser instance from bytes.

        Args:
            pdf_content: PDF content as bytes
            api_key: OpenAI API key
            description_model: Model to use for describing content
            visual_model: Model to use for layout dependency detection
            image_retries: Number of retries for image processing

        Returns:
            OpenAIParser instance
        """
        return cls(
            api_key=api_key,
            pdf_content=pdf_content,
            description_model=description_model or cls.DEFAULT_DESCRIPTION_MODEL,
            visual_model=visual_model or cls.DEFAULT_VISUAL_MODEL,
            image_retries=image_retries,
        )

    async def parse(self) -> ParsedPDFResult:
        try:
            document = pymupdf.open(stream=self.pdf_content)
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
            is_layout_dependent = await self._content_is_layout_dependent(page_as_image)
            if is_layout_dependent.content_is_layout_dependent:
                content = await self._describe_image_content(page_as_image)
                from_llm = True
            else:
                content = page_text
                from_llm = False
            return ParsedData(content=content, _from_llm=from_llm)
        except Exception as e:
            logging.error(f"Error parsing page {page_num}: {str(e)}")
            return ParsedData(
                content=f"Error processing page {page_num}. Fallback content: {page_text[:500]}...",
                _from_llm=False,
            )

    async def _describe_image_content(
        self,
        page_as_image: pymupdf.Pixmap,
    ) -> str:
        last_error = None
        for attempt in range(self.image_retries):
            try:
                openai = AsyncOpenAI(api_key=self.api_key)
                buffered = page_as_image.tobytes()
                img_str = base64.b64encode(buffered).decode("utf-8")

                system_prompt = """
                You are a helpful assistant that extracts and describes the content of PDF pages.
                Focus on extracting all the text content in a structured manner, preserving the exact text as written.
                Preserve the logical flow and hierarchy of information.
                
                For diagrams, charts, or visual elements:
                - Provide detailed descriptions of what they depict
                - Explain their purpose and relationship to the surrounding text
                - Include any labels, legends, or annotations visible in the diagram
                
                For document structure:
                - Maintain section headings and subheadings hierarchy
                - Preserve comparative elements (e.g., "versus", "compared to", "in contrast")
                - Clearly indicate when content is organized in columns, lists, or other structural formats
                
                If tables are present, reproduce them in a structured text format.
                Ensure the page content can be understood as a standalone document without referring to other pages.
                Ignore watermarks and page numbers.
                """

                response = await openai.responses.create(
                    input=[
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": "Extract and structure all the content from this PDF page.",
                                },
                                {
                                    "type": "input_image",
                                    "image_url": f"data:image/png;base64,{img_str}",
                                },
                            ],
                        },  # type: ignore
                    ],
                    model=self.description_model,
                )
                return response.output_text
            except Exception as e:
                last_error = e
                if attempt < self.image_retries - 1:
                    wait_time = (2**attempt) + (0.1 * attempt)
                    await asyncio.sleep(wait_time)

        # If all retries failed
        raise APIError(
            f"Failed to describe image after {self.image_retries} attempts: {str(last_error)}"
        )

    async def _content_is_layout_dependent(
        self, image: pymupdf.Pixmap
    ) -> VisualModelDecision:
        last_error = None
        for attempt in range(self.image_retries):
            try:
                openai = AsyncOpenAI(api_key=self.api_key)

                buffered = image.tobytes()
                img_str = base64.b64encode(buffered).decode("utf-8")

                system_prompt = """
                You need to determine if this PDF page has content that is layout-dependent.
                Layout-dependent content includes:
                - Tables, charts, or graphs
                - Complex formatting that affects meaning
                - Diagrams or flowcharts
                - Content arranged in columns that can't be linearly read
                - Math equations
                Return true if the content is layout dependent, false if it's just plain text that can be read linearly.
                """

                response = await openai.responses.parse(
                    input=[
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": "Is the content layout dependent? Respond with true or false.",
                                },
                                {
                                    "type": "input_image",
                                    "image_url": f"data:image/png;base64,{img_str}",
                                },
                            ],
                        },  # type: ignore
                    ],
                    model=self.visual_model,
                    text_format=VisualModelDecision,
                )
                return response.output_parsed or VisualModelDecision(
                    content_is_layout_dependent=False
                )
            except Exception as e:
                last_error = e
                if attempt < self.image_retries - 1:
                    wait_time = (2**attempt) + (0.1 * attempt)
                    await asyncio.sleep(wait_time)

        logging.error(f"Failed to determine layout dependency: {str(last_error)}")
        return VisualModelDecision(content_is_layout_dependent=True)
