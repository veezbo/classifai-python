"""Exceptions for ClassifAI client."""


class ClassifAIError(Exception):
    """Base exception for ClassifAI errors."""

    def __init__(self, message: str, status_code: int = None, response: dict = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.response = response


class AuthenticationError(ClassifAIError):
    """Raised when authentication fails (401)."""

    pass


class RateLimitError(ClassifAIError):
    """Raised when rate limit is exceeded (429)."""

    pass


class ValidationError(ClassifAIError):
    """Raised when request validation fails (400)."""

    pass


class NotFoundError(ClassifAIError):
    """Raised when resource is not found (404)."""

    pass
