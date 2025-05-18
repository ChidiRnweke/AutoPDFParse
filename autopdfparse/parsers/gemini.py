"""
Google Gemini-based PDF parser implementation.
"""

import asyncio
import importlib.util
from dataclasses import dataclass

from autopdfparse.default_prompts import (
    describe_image_system_prompt,
    layout_dependent_system_prompt,
)
from autopdfparse.exceptions import APIError, ModelError
from autopdfparse.models import VisualModelDecision
from autopdfparse.services import VisionService

from ..services.parser import PDFParser

GENAI_AVAILABLE = importlib.util.find_spec("google-genai") is not None


@dataclass
class GeminiVisionService(VisionService):
    """
    Implementation of VisionService using Google's Gemini vision capabilities.
    """

    api_key: str
    description_model: str
    visual_model: str
    describe_image_prompt: str
    layout_dependent_prompt: str
    retries: int = 3

    @classmethod
    def create(
        cls,
        api_key: str,
        description_model: str = "gemini-1.5-pro",
        visual_model: str = "gemini-1.5-flash",
        retries: int = 3,
        describe_image_prompt: str = describe_image_system_prompt,
        layout_dependent_prompt: str = layout_dependent_system_prompt,
    ) -> "GeminiVisionService":
        """
        Create a GeminiVisionService instance.

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
        if not GENAI_AVAILABLE:
            raise ModelError(
                "Google GenerativeAI package is not installed. Install it with 'pip install \"autopdfparse[gemini]\"'"
            )

        return cls(
            api_key=api_key,
            description_model=description_model,
            visual_model=visual_model,
            retries=retries,
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
            APIError: If the API call fails after retries
            ModelError: If Google GenerativeAI package is not installed
        """
        if not GENAI_AVAILABLE:
            raise ModelError(
                "Google GenerativeAI package is not installed. Install it with 'pip install \"autopdfparse[gemini]\"'"
            )

        from google import genai
        from google.genai import types

        last_error = None
        for attempt in range(self.retries):
            try:
                client = genai.Client(api_key=self.api_key)

                # Generate the response
                response = await client.aio.models.generate_content(
                    model=self.description_model,
                    contents=[
                        types.Part.from_bytes(
                            data=image.encode("utf-8"),
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
        Determine if the content in an image is layout-dependent using Google's Gemini model.

        Args:
            image: Image to analyze (base64 encoded)

        Returns:
            True if the content is layout-dependent, False otherwise

        Raises:
            APIError: If the API call fails after retries
            ModelError: If Google GenerativeAI package is not installed
        """
        if not GENAI_AVAILABLE:
            raise ModelError(
                "Google GenerativeAI package is not installed. Install it with 'pip install \"autopdfparse[gemini]\"'"
            )

        from google import genai
        from google.genai import types

        for attempt in range(self.retries):
            try:
                client = genai.Client(api_key=self.api_key)

                # Generate the response
                response = await client.aio.models.generate_content(
                    model=self.description_model,
                    contents=[
                        types.Part.from_bytes(
                            data=image.encode("utf-8"),
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
                if attempt < self.retries - 1:
                    wait_time = (2**attempt) + (0.1 * attempt)
                    await asyncio.sleep(wait_time)

        # Default to True on failure to ensure we don't miss layout-dependent content
        return True


class GeminiParser:
    """
    Factory class for creating PDF parsers that use Google Gemini's vision models.

    This class provides convenience methods for creating PDFParser instances
    that are configured to use Gemini's vision services.
    """

    @classmethod
    async def from_file(
        cls,
        file_path: str,
        api_key: str,
        description_model: str = "gemini-1.5-pro",
        visual_model: str = "gemini-1.5-flash",
        retries: int = 3,
    ) -> PDFParser:
        """
        Create a PDF parser from a file path using Google Gemini vision services.

        Args:
            file_path: Path to the PDF file
            api_key: Google API key
            description_model: Model to use for describing content
            visual_model: Model to use for layout dependency detection
            retries: Number of retries for API calls

        Returns:
            PDFParser instance configured with GeminiVisionService

        Raises:
            ModelError: If Google GenerativeAI package is not installed
        """
        if not GENAI_AVAILABLE:
            raise ModelError(
                "Google GenerativeAI package is not installed. Install it with 'pip install \"autopdfparse[gemini]\"'"
            )

        vision_service = GeminiVisionService.create(
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
        description_model: str = "gemini-1.5-pro",
        visual_model: str = "gemini-1.5-flash",
        retries: int = 3,
    ) -> PDFParser:
        """
        Create a PDF parser from bytes using Google Gemini vision services.

        Args:
            pdf_content: PDF content as bytes
            api_key: Google API key
            description_model: Model to use for describing content
            visual_model: Model to use for layout dependency detection
            retries: Number of retries for API calls

        Returns:
            PDFParser instance configured with GeminiVisionService

        Raises:
            ModelError: If Google GenerativeAI package is not installed
        """
        if not GENAI_AVAILABLE:
            raise ModelError(
                "Google GenerativeAI package is not installed. Install it with 'pip install \"autopdfparse[gemini]\"'"
            )

        vision_service = GeminiVisionService.create(
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
