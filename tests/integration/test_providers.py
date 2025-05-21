"""
Integration tests for different providers.

These tests require API keys to be set in environment variables:
- OPENAI_API_KEY for OpenAI tests
- ANTHROPIC_API_KEY for Anthropic tests
- GOOGLE_API_KEY for Google tests

Skip these tests if you don't want to make actual API calls.
"""

import os
from typing import Optional, cast

import pytest

from autopdfparse.providers import AnthropicParser, GeminiParser, OpenAIParser


def get_api_key(env_var: str) -> str | None:
    """Get API key from environment variable if it exists."""

    return os.environ.get(env_var)


@pytest.mark.asyncio
@pytest.mark.skipif(not get_api_key("OPENAI_API_KEY"), reason="OpenAI API key not set")
async def test_openai_integration(sample_pdf_bytes):
    """Test end-to-end parsing with OpenAI."""
    api_key: str = cast(str, get_api_key("OPENAI_API_KEY"))

    pdf_parser = OpenAIParser.get_parser(
        api_key=api_key,
    )

    result = await pdf_parser.parse_bytes(pdf_content=sample_pdf_bytes)
    assert result is not None
    assert len(result.pages) > 0
    assert all(page.content for page in result.pages)


@pytest.mark.asyncio
@pytest.mark.skipif(
    not get_api_key("ANTHROPIC_API_KEY"), reason="Anthropic API key not set"
)
async def test_anthropic_integration(sample_pdf_bytes):
    """Test end-to-end parsing with Anthropic."""
    api_key: str = cast(str, get_api_key("ANTHROPIC_API_KEY"))

    pdf_parser = AnthropicParser.get_parser(api_key=api_key)

    result = await pdf_parser.parse_bytes(pdf_content=sample_pdf_bytes)
    assert result is not None
    assert len(result.pages) > 0
    assert all(page.content for page in result.pages)


@pytest.mark.asyncio
@pytest.mark.skipif(not get_api_key("GOOGLE_API_KEY"), reason="Google API key not set")
async def test_gemini_integration(sample_pdf_bytes):
    """Test end-to-end parsing with Google Gemini."""
    api_key: str = cast(str, get_api_key("GOOGLE_API_KEY"))
    pdf_parser = GeminiParser.get_parser(api_key=api_key)

    result = await pdf_parser.parse_bytes(pdf_content=sample_pdf_bytes)
    assert result is not None
    assert len(result.pages) > 0
    assert all(page.content for page in result.pages)
