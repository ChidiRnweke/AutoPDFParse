import asyncio
import base64
from dataclasses import dataclass
from typing import Protocol

import pymupdf
from openai import AsyncOpenAI
from pydantic import BaseModel


@dataclass
class PDFPage:
    content: str
    page_number: int
    _from_llm: bool


@dataclass
class ParsedPDFResult:
    pages: list[PDFPage]


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
    model: str
    image_retries: int = 3

    async def parse(self) -> ParsedPDFResult:
        document = pymupdf.open(stream=self.pdf_content)
        images: list[pymupdf.Pixmap] = [page.get_pixmap() for page in document]  # type: ignore
        page_texts: list[str] = [page.get_text() for page in document]  # type: ignore
        if not images:
            return ParsedPDFResult(pages=[])

        async with asyncio.TaskGroup() as tg:
            tasks: list[asyncio.Task[ParsedData]] = []
            for i, (text, image) in enumerate(zip(page_texts, images)):
                tasks.append(tg.create_task(self._parse_page(text, image)))
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

    async def _parse_page(
        self, page_text: str, page_as_image: pymupdf.Pixmap
    ) -> ParsedData:
        is_layout_dependent = await self._content_is_layout_dependent(page_as_image)
        if is_layout_dependent.content_is_layout_dependent:
            content = await self._describe_image_content(page_as_image)
            from_llm = True
        else:
            content = page_text
            from_llm = False
        return ParsedData(content=content, _from_llm=from_llm)

    async def _describe_image_content(
        self,
        page_as_image: pymupdf.Pixmap,
    ) -> str:
        openai = AsyncOpenAI(api_key=self.api_key)
        buffered = page_as_image.tobytes()
        img_str = base64.b64encode(buffered).decode("utf-8")

        response = await openai.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that describes the content of an image.",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe the content of this image.",
                        },
                        {
                            "type": "image_url",
                            "image_url": f"data:image/png;base64,{img_str}",
                        },
                    ],
                },  # type: ignore
            ],
            model=self.model,
        )
        return response.choices[0].message.content or ""

    async def _content_is_layout_dependent(
        self, image: pymupdf.Pixmap
    ) -> VisualModelDecision:
        openai = AsyncOpenAI(api_key=self.api_key)

        buffered = image.tobytes()
        img_str = base64.b64encode(buffered).decode("utf-8")

        response = await openai.beta.chat.completions.parse(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that determines if the content of a PDF page is layout dependent.",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Is the content layout dependent?",
                        },
                        {
                            "type": "image_url",
                            "image_url": f"data:image/png;base64,{img_str}",
                        },
                    ],
                },  # type: ignore
            ],
            model=self.model,
            response_format=VisualModelDecision,
        )
        return response.choices[0].message.parsed or VisualModelDecision(
            content_is_layout_dependent=False
        )
