"""
Anthropic Claude-based PDF parser implementation.
"""

import asyncio
import importlib.util
from dataclasses import dataclass
from typing import ClassVar, Optional

from autopdfparse.default_prompts import (
    describe_image_system_prompt,
    layout_dependent_system_prompt,
)
from autopdfparse.exceptions import APIError, ModelError
from autopdfparse.models import VisualModelDecision
from autopdfparse.services import VisionService

from ..services.parser import PDFParser

# Check if Anthropic package is installed
ANTHROPIC_AVAILABLE = (
    importlib.util.find_spec("anthropic") is not None
    and importlib.util.find_spec("json-repair") is not None
)


@dataclass
class AnthropicVisionService(VisionService):
    """
    Implementation of VisionService using Anthropic Claude's vision capabilities.
    """

    api_key: str
    description_model: str
    visual_model: str
    describe_image_prompt: str
    layout_dependent_prompt: str
    retries: int = 3

    DEFAULT_DESCRIPTION_MODEL: ClassVar[str] = "claude-3-opus-20240229"
    DEFAULT_VISUAL_MODEL: ClassVar[str] = "claude-3-haiku-20240307"

    @classmethod
    def create(
        cls,
        api_key: str,
        description_model: Optional[str] = None,
        visual_model: Optional[str] = None,
        retries: int = 3,
        layout_dependent_prompt: str = layout_dependent_system_prompt,
        describe_image_prompt: str = describe_image_system_prompt,
    ) -> "AnthropicVisionService":
        """
        Create an AnthropicVisionService instance.

        Args:
            api_key: Anthropic API key
            description_model: Model to use for describing content
            visual_model: Model to use for layout dependency detection
            retries: Number of retries for API calls

        Returns:
            AnthropicVisionService instance

        Raises:
            ModelError: If Anthropic package is not installed
        """
        if not ANTHROPIC_AVAILABLE:
            raise ModelError(
                "Anthropic package is not installed. Install it with 'pip install \"autopdfparse[anthropic]\"'"
            )

        return cls(
            api_key=api_key,
            description_model=description_model or cls.DEFAULT_DESCRIPTION_MODEL,
            visual_model=visual_model or cls.DEFAULT_VISUAL_MODEL,
            retries=retries,
            layout_dependent_prompt=layout_dependent_prompt,
            describe_image_prompt=describe_image_prompt,
        )

    async def describe_image_content(self, image: str) -> str:
        """
        Describe the content of an image using Anthropic's Claude model.

        Args:
            image: Image to describe (base64 encoded)

        Returns:
            Text description of the image content

        Raises:
            APIError: If the API call fails after retries
            ModelError: If Anthropic package is not installed
        """
        if not ANTHROPIC_AVAILABLE:
            raise ModelError(
                "Anthropic package is not installed. Install it with 'pip install \"autopdfparse[anthropic]\"'"
            )

        from anthropic import AsyncAnthropic

        last_error = None
        for attempt in range(self.retries):
            try:
                client = AsyncAnthropic(api_key=self.api_key)

                message = await client.messages.create(
                    model=self.description_model,
                    max_tokens=64000,
                    system=self.describe_image_prompt,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Extract and structure all the content from this PDF page.",
                                },
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": image,
                                    },
                                },
                            ],
                        }
                    ],
                )
                return message.content[0].text  # type: ignore
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
        Determine if the content in an image is layout-dependent using Anthropic's Claude model.

        Args:
            image: Image to analyze (base64 encoded)

        Returns:
            True if the content is layout-dependent, False otherwise

        Raises:
            APIError: If the API call fails after retries
            ModelError: If Anthropic package is not installed
        """
        if not ANTHROPIC_AVAILABLE:
            raise ModelError(
                "Anthropic package is not installed. Install it with 'pip install \"autopdfparse[anthropic]\"'"
            )

        import json_repair
        from anthropic import AsyncAnthropic

        for attempt in range(self.retries):
            try:
                client = AsyncAnthropic(api_key=self.api_key)

                message = await client.messages.create(
                    model=self.visual_model,
                    max_tokens=100,
                    system=f"{self.layout_dependent_prompt}",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": image,
                                    },
                                },
                                {
                                    "type": "text",
                                    "text": "Is the content layout dependent? Respond with a JSON object containing a boolean field 'content_is_layout_dependent'. Example: {\"content_is_layout_dependent\": true}",
                                },
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": '{"content_is_layout_dependent": ',
                        },
                    ],
                )

                text_content = message.content[0].text  # type: ignore
                result = VisualModelDecision(**json_repair.loads((text_content)))  # type: ignore
                return result.content_is_layout_dependent

            except Exception:
                if attempt < self.retries - 1:
                    wait_time = (2**attempt) + (0.1 * attempt)
                    await asyncio.sleep(wait_time)

        # Default to True on failure to ensure we don't miss layout-dependent content
        return True


class AnthropicParser:
    """
    Factory class for creating PDF parsers that use Anthropic Claude's vision models.

    This class provides convenience methods for creating PDFParser instances
    that are configured to use Anthropic's vision services.
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
        Create a PDF parser from a file path using Anthropic Claude vision services.

        Args:
            file_path: Path to the PDF file
            api_key: Anthropic API key
            description_model: Model to use for describing content
            visual_model: Model to use for layout dependency detection
            retries: Number of retries for API calls

        Returns:
            PDFParser instance configured with AnthropicVisionService

        Raises:
            ModelError: If Anthropic package is not installed
        """
        if not ANTHROPIC_AVAILABLE:
            raise ModelError(
                "Anthropic package is not installed. Install it with 'pip install \"autopdfparse[anthropic]\"'"
            )

        vision_service = AnthropicVisionService.create(
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
        Create a PDF parser from bytes using Anthropic Claude vision services.

        Args:
            pdf_content: PDF content as bytes
            api_key: Anthropic API key
            description_model: Model to use for describing content
            visual_model: Model to use for layout dependency detection
            retries: Number of retries for API calls

        Returns:
            PDFParser instance configured with AnthropicVisionService

        Raises:
            ModelError: If Anthropic package is not installed
        """
        if not ANTHROPIC_AVAILABLE:
            raise ModelError(
                "Anthropic package is not installed. Install it with 'pip install \"autopdfparse[anthropic]\"'"
            )

        vision_service = AnthropicVisionService.create(
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
