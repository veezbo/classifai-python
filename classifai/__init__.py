"""ClassifAI Python Client Library

A simple client for the ClassifAI API - a self-improving classification API for text and images.
"""

from .client import ClassifAI
from .exceptions import (
    ClassifAIError,
    AuthenticationError,
    RateLimitError,
    ValidationError,
    NotFoundError,
)

__version__ = "0.1.0"
__all__ = [
    "ClassifAI",
    "ClassifAIError",
    "AuthenticationError",
    "RateLimitError",
    "ValidationError",
    "NotFoundError",
]
