"""
Unit tests for the PDF parser using local vision service.
"""

import os
import tempfile

import pytest

from autopdfparse.exceptions import PDFParsingError
from autopdfparse.services import PDFParser


@pytest.mark.asyncio
async def test_parser_create_from_file(sample_pdf_bytes, local_vision_service):
    """Test creating a parser from a file path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
        temp.write(sample_pdf_bytes)
        temp_path = temp.name

    try:
        parser = PDFParser(vision_service=local_vision_service)
        await parser.parse_file(file_path=temp_path)
        assert parser is not None
    finally:
        os.unlink(temp_path)


@pytest.mark.asyncio
async def test_parser_create_file_not_found(local_vision_service):
    """Test error handling when file is not found."""
    with pytest.raises(PDFParsingError) as excinfo:
        await PDFParser(vision_service=local_vision_service).parse_file(
            file_path="non_existent_file.pdf"
        )
    assert "File not found" in str(excinfo.value)


@pytest.mark.asyncio
async def test_parser_parse_basic_pdf(sample_pdf_bytes, local_vision_service):
    """Test parsing a basic PDF document."""
    parser = PDFParser(vision_service=local_vision_service)
    result = await parser.parse_bytes(pdf_content=sample_pdf_bytes)

    assert result is not None
    assert len(result.pages) == 1
    assert result.pages[0].page_number == 1
    assert isinstance(result.pages[0].content, str)
    assert result.get_all_content() == result.pages[0].content


@pytest.mark.asyncio
async def test_parser_parse_multipage_pdf(multipage_pdf_bytes, local_vision_service):
    """Test parsing a multi-page PDF document."""
    parser = PDFParser(vision_service=local_vision_service)
    result = await parser.parse_bytes(pdf_content=multipage_pdf_bytes)

    assert result is not None
    assert len(result.pages) == 2
    assert result.pages[0].page_number == 1
    assert result.pages[1].page_number == 2
    assert (
        result.get_all_content()
        == f"{result.pages[0].content}\n\n{result.pages[1].content}"
    )


@pytest.mark.asyncio
async def test_parser_with_layout_dependent_content(
    sample_pdf_bytes,
    layout_dependent_vision_service,
):
    """Test parsing when content is determined to be layout-dependent."""
    parser = PDFParser(vision_service=layout_dependent_vision_service)
    result = await parser.parse_bytes(pdf_content=sample_pdf_bytes)

    assert result.pages[0]._from_llm is True
    assert "This page contains" in result.pages[0].content


@pytest.mark.asyncio
async def test_parser_with_layout_independent_content(
    sample_pdf_bytes, layout_independent_vision_service
):
    """Test parsing when content is determined to be layout-independent."""
    parser = PDFParser(vision_service=layout_independent_vision_service)
    result = await parser.parse_bytes(pdf_content=sample_pdf_bytes)

    assert result.pages[0]._from_llm is False
    assert "This is a test PDF document" in result.pages[0].content


@pytest.mark.asyncio
async def test_parser_with_empty_pdf(empty_pdf_bytes, local_vision_service):
    """Test parsing an empty PDF."""
    parser = PDFParser(vision_service=local_vision_service)
    result = await parser.parse_bytes(pdf_content=empty_pdf_bytes)

    assert result is not None
    assert len(result.pages) == 1
    assert result.pages[0].content.strip() == ""
