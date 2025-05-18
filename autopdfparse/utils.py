"""
Utility functions for the AutoPDFParse package.
"""

import asyncio
import logging
from typing import Any, Awaitable, Callable, Optional, TypeVar

T = TypeVar("T")


async def retry_async(
    func: Callable[..., Awaitable[T]],
    retries: int,
    initial_delay: float = 0.1,
    max_delay: float = 10.0,
    backoff_factor: float = 2.0,
    logger: Optional[logging.Logger] = None,
    *args: Any,
    **kwargs: Any,
) -> T:
    """
    Retry an async function with exponential backoff.

    Args:
        func: The async function to retry
        retries: Maximum number of retries
        initial_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        backoff_factor: Backoff multiplier
        logger: Logger to use for logging errors
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function

    Returns:
        The result of the function

    Raises:
        The last exception raised by the function
    """
    last_exception = None
    delay = initial_delay

    for attempt in range(retries + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            if logger:
                logger.warning(f"Attempt {attempt + 1}/{retries + 1} failed: {str(e)}")
            if attempt < retries:
                await asyncio.sleep(min(delay, max_delay))
                delay *= backoff_factor
            else:
                break

    if last_exception:
        raise last_exception

    # This should never happen, but makes type checking happy
    raise RuntimeError("Unexpected error in retry_async")
