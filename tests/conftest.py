"""
Fixtures for AutoPDFParse tests.
"""

import pymupdf
import pytest

from .utils import LocalVisionService


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    """Return a simple in-memory PDF for testing."""
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)  # type: ignore
    page.insert_text((72, 72), "This is a test PDF document.")
    return doc.write()


@pytest.fixture
def sample_pdf_with_table_bytes() -> bytes:
    """Return an in-memory PDF with a simple table for testing."""
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)  # type: ignore
    page.insert_text((72, 72), "This is a test PDF document with a table:")

    # Create a simple table
    table_y = 100
    for row in range(3):
        for col in range(3):
            x = 72 + col * 100
            y = table_y + row * 30
            page.insert_text((x, y), f"Cell {row},{col}")

    return doc.write()


@pytest.fixture
def multipage_pdf_bytes() -> bytes:
    """Return a simple in-memory multi-page PDF for testing."""
    doc = pymupdf.open()

    # Page 1
    page = doc.new_page(width=595, height=842)  # type: ignore
    page.insert_text((72, 72), "This is page 1 of the test PDF document.")

    # Page 2
    page = doc.new_page(width=595, height=842)  # type: ignore
    page.insert_text((72, 72), "This is page 2 of the test PDF document.")

    return doc.write()


@pytest.fixture
def empty_pdf_bytes() -> bytes:
    """Return an empty PDF for testing."""
    doc = pymupdf.open()
    doc.new_page(width=595, height=842)  # type: ignore
    return doc.write()


@pytest.fixture
def corrupted_pdf_bytes() -> bytes:
    """Return corrupted PDF bytes for testing error handling."""
    valid_pdf = pymupdf.open()
    valid_pdf.new_page()  # type: ignore
    pdf_bytes = valid_pdf.write()
    # Corrupt the PDF by truncating it
    return pdf_bytes[: len(pdf_bytes) // 2]


@pytest.fixture
def local_vision_service() -> LocalVisionService:
    """Return a local vision service instance for testing."""
    return LocalVisionService()


@pytest.fixture
def local_vision_service_with_responses() -> LocalVisionService:
    """Return a local vision service with predefined responses."""
    # Define some deterministic responses
    responses: dict[str, str] = {}

    # These will be filled with actual hash values during tests
    # when we know the image hash values

    return LocalVisionService(content_responses=responses)


@pytest.fixture
def layout_dependent_vision_service() -> LocalVisionService:
    """Return a vision service that always reports layout dependency."""
    return LocalVisionService(layout_dependent_pattern=".*")


@pytest.fixture
def layout_independent_vision_service() -> LocalVisionService:
    """Return a vision service that never reports layout dependency."""
    return LocalVisionService(layout_dependent_pattern="^$")  # Never matches
