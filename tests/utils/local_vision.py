"""
Local vision service implementation for testing without API calls.
"""

import hashlib
import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LocalVisionService:
    """
    A local vision service that returns deterministic responses without API calls.

    This service is intended for testing purposes only and simulates the behavior
    of cloud vision services without making actual API calls.
    """

    layout_dependent_pattern: str | None = None
    content_responses: dict[str, str] = field(default_factory=lambda: {})

    def _hash_image(self, image: str) -> str:
        """Generate a deterministic hash from an image string."""
        return hashlib.md5(image.encode()).hexdigest()

    async def describe_image_content(self, image: str) -> str:
        """
        Return a deterministic description based on the image content.

        Args:
            image: Image content as a string (base64, etc.)

        Returns:
            A generated description of the image content
        """
        image_hash = self._hash_image(image)

        # Use predefined response if available
        if image_hash in self.content_responses:
            return self.content_responses[image_hash]

        # Generate deterministic content based on the hash
        image_num = int(image_hash, 16) % 1000
        content_types = ["table", "paragraph", "form", "invoice", "chart"]
        content_type = content_types[image_num % len(content_types)]

        return (
            f"This page contains a {content_type} with sample data. The content was "
            f"extracted from image with hash {image_hash[:8]}."
        )

    async def is_layout_dependent(self, image: str) -> bool:
        """
        Determine if content is layout-dependent based on image hash.

        Args:
            image: Image content as a string (base64, etc.)

        Returns:
            True if the content should be treated as layout-dependent
        """
        image_hash = self._hash_image(image)

        # Check against pattern if provided
        if self.layout_dependent_pattern and re.search(
            self.layout_dependent_pattern, image_hash
        ):
            return True

        # Default: even hashes are layout-dependent
        return int(image_hash, 16) % 2 == 0


@dataclass
class LocalSyncVisionService:
    """Synchronous version of LocalVisionService for sync API testing."""

    layout_dependent_pattern: str | None = None
    content_responses: dict[str, str] = field(default_factory=lambda: {})

    def _hash_image(self, image: str) -> str:
        """Generate a deterministic hash from an image string."""
        return hashlib.md5(image.encode()).hexdigest()

    def describe_image_content(self, image: str) -> str:
        """Return a deterministic description based on the image content."""
        image_hash = self._hash_image(image)

        # Use predefined response if available
        if image_hash in self.content_responses:
            return self.content_responses[image_hash]

        # Generate deterministic content based on the hash
        image_num = int(image_hash, 16) % 1000
        content_types = ["table", "paragraph", "form", "invoice", "chart"]
        content_type = content_types[image_num % len(content_types)]

        return (
            f"This page contains a {content_type} with sample data. The content was "
            f"extracted from image with hash {image_hash[:8]}."
        )

    def is_layout_dependent(self, image: str) -> bool:
        """Determine if content is layout-dependent based on image hash."""
        image_hash = self._hash_image(image)

        # Check against pattern if provided
        if self.layout_dependent_pattern and re.search(
            self.layout_dependent_pattern, image_hash
        ):
            return True

        # Default: even hashes are layout-dependent
        return int(image_hash, 16) % 2 == 0
