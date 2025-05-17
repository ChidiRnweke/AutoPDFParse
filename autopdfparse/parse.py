import asyncio
import base64
import logging
from dataclasses import dataclass
from typing import ClassVar, Optional, Protocol

import pymupdf
from openai import AsyncOpenAI
from pydantic import BaseModel


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


@dataclass
class PDFPage:
    content: str
    page_number: int
    _from_llm: bool


@dataclass
class ParsedPDFResult:
    pages: list[PDFPage]

    def get_all_content(self) -> str:
        """Returns all page content concatenated into a single string."""
        return "\n\n".join(page.content for page in self.pages)


@dataclass
class ParsedData:
    content: str
    _from_llm: bool


class PDFParser(Protocol):
    async def parse(self) -> ParsedPDFResult: ...


class VisualModelDecision(BaseModel):
    content_is_layout_dependent: bool


@dataclass
class OpenAIParser(PDFParser):
    api_key: str
    pdf_content: bytes
    description_model: str
    visual_model: str
    image_retries: int = 3

    DEFAULT_DESCRIPTION_MODEL: ClassVar[str] = "gpt-4.1"
    DEFAULT_VISUAL_MODEL: ClassVar[str] = "gpt-4.1-mini"

    @classmethod
    async def from_file(
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
    async def from_bytes(
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
                Focus on extracting all the text content in a structured manner.
                Preserve the logical flow and hierarchy of information.
                If tables are present, reproduce them in a structured text format.
                Ignore watermarks and page numbers.
                """

                response = await openai.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Extract and structure all the content from this PDF page.",
                                },
                                {
                                    "type": "image_url",
                                    "image_url": f"data:image/png;base64,{img_str}",
                                },
                            ],
                        },  # type: ignore
                    ],
                    model=self.description_model,
                )
                return response.choices[0].message.content or ""
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

                response = await openai.beta.chat.completions.parse(
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Is the content layout dependent? Respond with true or false.",
                                },
                                {
                                    "type": "image_url",
                                    "image_url": f"data:image/png;base64,{img_str}",
                                },
                            ],
                        },  # type: ignore
                    ],
                    model=self.visual_model,
                    response_format=VisualModelDecision,
                )
                return response.choices[0].message.parsed or VisualModelDecision(
                    content_is_layout_dependent=False
                )
            except Exception as e:
                last_error = e
                if attempt < self.image_retries - 1:
                    wait_time = (2**attempt) + (0.1 * attempt)
                    await asyncio.sleep(wait_time)

        logging.error(f"Failed to determine layout dependency: {str(last_error)}")
        return VisualModelDecision(content_is_layout_dependent=True)
