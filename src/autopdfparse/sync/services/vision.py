"""
Synchronous vision service interfaces.
"""

from typing import Protocol


class VisionService(Protocol):
    """Interface for synchronous vision services that can analyze PDF content."""

    def describe_image_content(self, image: str) -> str:
        """
        Describe the content of an image.

        Args:
            image: Image to describe

        Returns:
            Text description of the image content
        """
        ...

    def is_layout_dependent(self, image: str) -> bool:
        """
        Determine if the content in an image is layout-dependent.

        Args:
            image: Image to analyze

        Returns:
            True if the content is layout-dependent, False otherwise
        """
        ...
