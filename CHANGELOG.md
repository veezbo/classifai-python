# Changelog

All notable changes to the ClassifAI Python client will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-01-XX

### Added
- Initial release of ClassifAI Python client
- `ClassifAI` client class with API key authentication
- `classify()` method for text and image classification
- `classify_from_files()` convenience method with automatic file and URL handling
- `submit_feedback()` method for ground truth feedback
- `get_project_stats()` method for project analytics
- `health_check()` method for API health status
- Support for mixed text and image content (analyzed jointly)
- Automatic base64 encoding for images
- Custom exception classes: `ClassifAIError`, `AuthenticationError`, `RateLimitError`, `ValidationError`, `NotFoundError`
- Comprehensive examples for basic usage, image classification, and real-world use cases
- Full documentation in README.md

### Features
- Classify text strings, lists of text, or mixed content
- Load images from local files or download from URLs
- Automatic label inference from descriptions
- Project-based organization with reusable label sets
- Ground truth feedback loop for improving accuracy
- Detailed error handling and informative exceptions

[0.1.0]: https://github.com/yourusername/classifai-python/releases/tag/v0.1.0
