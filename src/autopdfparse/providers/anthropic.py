"""
Anthropic Claude-based PDF parser implementation.
"""

import importlib.util
from asyncio import Semaphore
from dataclasses import dataclass
from typing import ClassVar

from autopdfparse.config import Config
from autopdfparse.default_prompts import (
    describe_image_system_prompt,
    layout_dependent_system_prompt,
)
from autopdfparse.exceptions import APIError, ModelError
from autopdfparse.models import VisualModelDecision
from autopdfparse.services import PDFParser, VisionService

# Check if Anthropic package is installed
ANTHROPIC_AVAILABLE = (
    importlib.util.find_spec("anthropic") is not None
    and importlib.util.find_spec("json_repair") is not None
)


@dataclass
class AnthropicParser(VisionService):
    """
    Implementation of VisionService using Anthropic Claude's vision capabilities.
    """

    api_key: str
    description_model: str
    visual_model: str
    describe_image_prompt: str
    layout_dependent_prompt: str
    _semaphore: ClassVar[Semaphore] = Semaphore(Config.MAX_CONCURRENT_REQUESTS)

    @classmethod
    def get_parser(
        cls,
        api_key: str,
        description_model: str = "claude-3-7-sonnet-latest",
        visual_model: str = "claude-3-5-haiku-latest",
        describe_image_prompt: str = describe_image_system_prompt,
        layout_dependent_prompt: str = layout_dependent_system_prompt,
    ) -> PDFParser:
        """
        Create a PDF parser instance using Anthropic's vision capabilities.

        Args:
            api_key: Anthropic API key
            description_model: Model to use for describing content
            visual_model: Model to use for layout dependency detection

        Returns:
            PDFParser instance
        """
        return PDFParser(
            vision_service=cls._create_vision_service(
                api_key=api_key,
                description_model=description_model,
                visual_model=visual_model,
                describe_image_prompt=describe_image_prompt,
                layout_dependent_prompt=layout_dependent_prompt,
            )
        )

    @classmethod
    def _create_vision_service(
        cls,
        api_key: str,
        description_model: str = "claude-3-7-sonnet-latest",
        visual_model: str = "claude-3-5-haiku-latest",
        layout_dependent_prompt: str = layout_dependent_system_prompt,
        describe_image_prompt: str = describe_image_system_prompt,
    ) -> "AnthropicParser":
        """
        Create an AnthropicParser instance.

        Args:
            api_key: Anthropic API key
            description_model: Model to use for describing content
            visual_model: Model to use for layout dependency detection

        Returns:
            AnthropicParser instance

        Raises:
            ModelError: If Anthropic package is not installed
        """
        if not ANTHROPIC_AVAILABLE:
            raise ModelError(
                "Anthropic package is not installed. Install it with 'pip install \"autopdfparse[anthropic]\"'"
            )

        return cls(
            api_key=api_key,
            description_model=description_model,
            visual_model=visual_model,
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
            APIError: If the API call fails
            ModelError: If Anthropic package is not installed
        """
        if not ANTHROPIC_AVAILABLE:
            raise ModelError(
                "Anthropic package is not installed. Install it with 'pip install \"autopdfparse[anthropic]\"'"
            )

        try:
            from anthropic import AsyncAnthropic

            client = AsyncAnthropic(api_key=self.api_key)

            async with self._semaphore:
                message = await client.messages.create(
                    model=self.description_model,
                    max_tokens=1024,
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
            raise APIError(f"Failed to describe image: {str(e)}")

    async def is_layout_dependent(self, image: str) -> bool:
        """
        Determine if the content in an image is layout-dependent using Anthropic's Claude model.

        Args:
            image: Image to analyze (base64 encoded)

        Returns:
            True if the content is layout-dependent, False otherwise

        Raises:
            APIError: If the API call fails
            ModelError: If Anthropic package is not installed
        """
        if not ANTHROPIC_AVAILABLE:
            raise ModelError(
                "Anthropic package is not installed. Install it with 'pip install \"autopdfparse[anthropic]\"'"
            )

        try:
            import json_repair
            from anthropic import AsyncAnthropic

            client = AsyncAnthropic(api_key=self.api_key)

            async with self._semaphore:
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
                result = VisualModelDecision(**json_repair.loads(text_content))  # type: ignore
                return result.content_is_layout_dependent
        except Exception:
            # Default to True on failure to ensure we don't miss layout-dependent content
            return True
