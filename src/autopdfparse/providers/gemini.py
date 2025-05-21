"""
Google Gemini-based PDF parser implementation.
"""

import base64
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
from autopdfparse.services import VisionService
from autopdfparse.services.parser import PDFParser

GEMINI_AVAILABLE = importlib.util.find_spec("google") is not None


@dataclass
class GeminiParser(VisionService):
    """
    Implementation of VisionService using Google's Gemini vision capabilities.
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
        description_model: str = "gemini-1.5-flash",
        visual_model: str = "gemini-1.5-flash",
        describe_image_prompt: str = describe_image_system_prompt,
        layout_dependent_prompt: str = layout_dependent_system_prompt,
    ) -> PDFParser:
        """
        Create a PDF parser instance using Google's Gemini vision capabilities.

        Args:
            api_key: Google API key
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
        description_model: str = "gemini-1.5-flash",
        visual_model: str = "gemini-1.5-flash",
        describe_image_prompt: str = describe_image_system_prompt,
        layout_dependent_prompt: str = layout_dependent_system_prompt,
    ) -> "GeminiParser":
        """
        Create a GeminiParser instance.

        Args:
            api_key: Google API key
            description_model: Model to use for describing content
            visual_model: Model to use for layout dependency detection
            retries: Number of retries for API calls
            describe_image_prompt: System prompt for describing images
            layout_dependent_prompt: System prompt for determining layout dependency

        Returns:
            GeminiVisionService instance

        Raises:
            ModelError: If Google GenerativeAI package is not installed
        """
        if not GEMINI_AVAILABLE:
            raise ModelError(
                "Google GenerativeAI package is not installed. Install it with 'pip install \"autopdfparse[gemini]\"'"
            )

        return cls(
            api_key=api_key,
            description_model=description_model,
            visual_model=visual_model,
            describe_image_prompt=describe_image_prompt,
            layout_dependent_prompt=layout_dependent_prompt,
        )

    async def describe_image_content(self, image: str) -> str:
        """
        Describe the content of an image using Google's Gemini model.

        Args:
            image: Image to describe (base64 encoded)

        Returns:
            Text description of the image content

        Raises:
            APIError: If the API call fails
            ModelError: If Google GenerativeAI package is not installed
        """
        if not GEMINI_AVAILABLE:
            raise ModelError(
                "Google GenerativeAI package is not installed. Install it with 'pip install \"autopdfparse[gemini]\"'"
            )

        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self.api_key)

            # Use semaphore to limit concurrent requests
            async with self._semaphore:
                # Generate the response
                response = await client.aio.models.generate_content(
                    model=self.description_model,
                    contents=[
                        types.Part.from_bytes(
                            data=base64.b64decode(image),
                            mime_type="image/png",
                        ),
                        "Extract and structure all the content from this PDF page.",
                    ],
                    config=types.GenerateContentConfig(
                        system_instruction=self.describe_image_prompt,
                    ),
                )

                return response.text or ""
        except Exception as e:
            raise APIError(f"Failed to describe image: {str(e)}")

    async def is_layout_dependent(self, image: str) -> bool:
        """
        Determine if the content in an image is layout-dependent using Google's Gemini model.

        Args:
            image: Image to analyze (base64 encoded)

        Returns:
            True if the content is layout-dependent, False otherwise

        Raises:
            APIError: If the API call fails
            ModelError: If Google GenerativeAI package is not installed
        """
        if not GEMINI_AVAILABLE:
            raise ModelError(
                "Google-genai package is not installed. Install it with 'pip install \"autopdfparse[gemini]\"'"
            )

        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self.api_key)

            async with self._semaphore:
                response = await client.aio.models.generate_content(
                    model=self.description_model,
                    contents=[
                        types.Part.from_bytes(
                            data=base64.b64decode(image),
                            mime_type="image/png",
                        ),
                        "Is the content layout dependent? Respond with true or false",
                    ],
                    config=types.GenerateContentConfig(
                        system_instruction=self.layout_dependent_prompt,
                        response_mime_type="application/json",
                        response_schema=VisualModelDecision,
                    ),
                )

                return response.parsed.content_is_layout_dependent  # type: ignore
        except Exception:
            # Default to True on failure to ensure we don't miss layout-dependent content
            return True
