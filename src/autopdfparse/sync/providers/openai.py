"""
Synchronous OpenAI-based PDF parser implementation.
"""

import importlib.util
from dataclasses import dataclass

from autopdfparse.default_prompts import (
    describe_image_system_prompt,
    layout_dependent_system_prompt,
)
from autopdfparse.exceptions import APIError, ModelError
from autopdfparse.models import VisualModelDecision
from autopdfparse.sync.services import PDFParser, VisionService

OPENAI_AVAILABLE = importlib.util.find_spec("openai") is not None


@dataclass
class OpenAIParser(VisionService):
    """
    Synchronous implementation of VisionService using OpenAI's vision capabilities.
    """

    api_key: str
    description_model: str
    visual_model: str
    describe_image_prompt: str
    layout_dependent_prompt: str

    @classmethod
    def get_parser(
        cls,
        api_key: str,
        description_model: str = "gpt-4.1",
        visual_model: str = "gpt-4.1-mini",
        describe_image_prompt: str = describe_image_system_prompt,
        layout_dependent_prompt: str = layout_dependent_system_prompt,
    ) -> PDFParser:
        """
        Create a PDF parser instance using OpenAI's vision capabilities.

        Args:
            api_key: OpenAI API key
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
        description_model: str = "gpt-4.1",
        visual_model: str = "gpt-4.1-mini",
        describe_image_prompt: str = describe_image_system_prompt,
        layout_dependent_prompt: str = layout_dependent_system_prompt,
    ) -> "OpenAIParser":
        """
        Create an OpenAIParser instance.

        Args:
            api_key: OpenAI API key
            description_model: Model to use for describing content
            visual_model: Model to use for layout dependency detection
            retries: Number of retries for API calls

        Returns:
            OpenAIParser instance

        Raises:
            ModelError: If OpenAI package is not installed
        """
        if not OPENAI_AVAILABLE:
            raise ModelError(
                "OpenAI package is not installed. Install it with 'pip install \"autopdfparse[openai]\"'"
            )

        return cls(
            api_key=api_key,
            description_model=description_model,
            visual_model=visual_model,
            describe_image_prompt=describe_image_prompt,
            layout_dependent_prompt=layout_dependent_prompt,
        )

    def describe_image_content(self, image: str) -> str:
        """
        Describe the content of an image using OpenAI's vision model.

        Args:
            image: Image to describe

        Returns:
            Text description of the image content

        Raises:
            APIError: If the API call fails
            ModelError: If OpenAI package is not installed
        """
        if not OPENAI_AVAILABLE:
            raise ModelError(
                "OpenAI package is not installed. Install it with 'pip install \"autopdfparse[openai]\"'"
            )

        try:
            from openai import OpenAI

            openai = OpenAI(api_key=self.api_key)
            response = openai.responses.create(
                input=[
                    {
                        "role": "system",
                        "content": self.describe_image_prompt,
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": "Extract and structure all the content from this image.",
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
            raise APIError(f"Failed to describe image: {str(e)}")

    def is_layout_dependent(self, image: str) -> bool:
        """
        Determine if the content in an image is layout-dependent using OpenAI's vision model.

        Args:
            image: Image to analyze

        Returns:
            True if the content is layout-dependent, False otherwise

        Raises:
            APIError: If the API call fails
            ModelError: If OpenAI package is not installed
        """
        if not OPENAI_AVAILABLE:
            raise ModelError(
                "OpenAI package is not installed. Install it with 'pip install \"autopdfparse[openai]\"'"
            )

        from openai import OpenAI

        try:
            openai = OpenAI(api_key=self.api_key)
            response = openai.responses.parse(
                input=[
                    {
                        "role": "system",
                        "content": self.layout_dependent_prompt,
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

            return (
                response.output_parsed.content_is_layout_dependent
                if response.output_parsed
                else False
            )
        except Exception:
            # Default to True on failure to ensure we don't miss layout-dependent content
            return True
