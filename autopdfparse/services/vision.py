"""
Vision service interfaces and implementations.
"""

import asyncio
from dataclasses import dataclass
from typing import ClassVar, Optional, Protocol

from openai import AsyncOpenAI

from ..exceptions import APIError
from ..models import VisualModelDecision


class VisionService(Protocol):
    """Interface for vision services that can analyze PDF content."""

    async def describe_image_content(self, image: str) -> str:
        """
        Describe the content of an image.

        Args:
            image: Image to describe

        Returns:
            Text description of the image content
        """
        ...

    async def is_layout_dependent(self, image: str) -> bool:
        """
        Determine if the content in an image is layout-dependent.

        Args:
            image: Image to analyze

        Returns:
            True if the content is layout-dependent, False otherwise
        """
        ...


@dataclass
class OpenAIVisionService(VisionService):
    """
    Implementation of VisionService using OpenAI's vision capabilities.
    """

    api_key: str
    description_model: str
    visual_model: str
    retries: int = 3

    # Default models to use
    DEFAULT_DESCRIPTION_MODEL: ClassVar[str] = "gpt-4.1"
    DEFAULT_VISUAL_MODEL: ClassVar[str] = "gpt-4.1-mini"

    @classmethod
    def create(
        cls,
        api_key: str,
        description_model: Optional[str] = None,
        visual_model: Optional[str] = None,
        retries: int = 3,
    ) -> "OpenAIVisionService":
        """
        Create an OpenAIVisionService instance.

        Args:
            api_key: OpenAI API key
            description_model: Model to use for describing content
            visual_model: Model to use for layout dependency detection
            retries: Number of retries for API calls

        Returns:
            OpenAIVisionService instance
        """
        return cls(
            api_key=api_key,
            description_model=description_model or cls.DEFAULT_DESCRIPTION_MODEL,
            visual_model=visual_model or cls.DEFAULT_VISUAL_MODEL,
            retries=retries,
        )

    async def describe_image_content(self, image: str) -> str:
        """
        Describe the content of an image using OpenAI's vision model.

        Args:
            image: Image to describe

        Returns:
            Text description of the image content

        Raises:
            APIError: If the API call fails after retries
        """
        last_error = None
        for attempt in range(self.retries):
            try:
                openai = AsyncOpenAI(api_key=self.api_key)

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
                                    "image_url": f"data:image/png;base64,{image}",
                                },
                            ],
                        },  # type: ignore
                    ],
                    model=self.description_model,
                )
                return response.output_text
            except Exception as e:
                last_error = e
                if attempt < self.retries - 1:
                    wait_time = (2**attempt) + (0.1 * attempt)
                    await asyncio.sleep(wait_time)

        # If all retries failed
        raise APIError(
            f"Failed to describe image after {self.retries} attempts: {str(last_error)}"
        )

    async def is_layout_dependent(self, image: str) -> bool:
        """
        Determine if the content in an image is layout-dependent using OpenAI's vision model.

        Args:
            image: Image to analyze

        Returns:
            True if the content is layout-dependent, False otherwise

        Raises:
            APIError: If the API call fails after retries
        """
        for attempt in range(self.retries):
            try:
                openai = AsyncOpenAI(api_key=self.api_key)

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
                                    "image_url": f"data:image/png;base64,{image}",
                                },
                            ],
                        },  # type: ignore
                    ],
                    model=self.visual_model,
                    text_format=VisualModelDecision,
                )
                result = response.output_parsed or VisualModelDecision(
                    content_is_layout_dependent=False
                )
                return result.content_is_layout_dependent
            except Exception:
                if attempt < self.retries - 1:
                    wait_time = (2**attempt) + (0.1 * attempt)
                    await asyncio.sleep(wait_time)

        return True
