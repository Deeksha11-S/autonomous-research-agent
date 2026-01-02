import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from loguru import logger
import hashlib
import re


class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        return super().default(obj)


def setup_logging():
    """Configure structured logging"""
    logger.add(
        "logs/app_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="30 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}:{function}:{line} | {message}",
        level="INFO"
    )
    return logger


def validate_url(url: str) -> bool:
    """Validate URL format"""
    pattern = re.compile(
        r'^(https?://)?'  # protocol
        r'(([A-Z0-9][A-Z0-9_-]*)(\.[A-Z0-9][A-Z0-9_-]*)+)'  # domain
        r'(:\d+)?'  # port
        r'(/.*)?$', re.IGNORECASE)
    return bool(pattern.match(url))


def generate_id(content: str) -> str:
    """Generate deterministic ID from content"""
    return hashlib.md5(content.encode()).hexdigest()[:12]


async def retry_async(func, max_retries=3, delay=1, backoff=2, exceptions=None):
    """Retry async function with exponential backoff"""
    if exceptions is None:
        exceptions = (Exception,)

    for attempt in range(max_retries):
        try:
            return await func()
        except exceptions as e:
            if attempt == max_retries - 1:
                raise
            wait_time = delay * (backoff ** attempt)
            logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
            await asyncio.sleep(wait_time)


def sanitize_text(text: str) -> str:
    """Sanitize text for LLM input"""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove control characters
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    # Limit length
    max_length = 100000  # ~25k tokens
    if len(text) > max_length:
        text = text[:max_length] + "... [TRUNCATED]"
    return text.strip()