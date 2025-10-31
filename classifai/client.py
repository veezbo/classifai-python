"""ClassifAI client for making API requests."""

import base64
from typing import List, Dict, Optional, Union
from pathlib import Path

import requests

from .exceptions import (
    ClassifAIError,
    AuthenticationError,
    RateLimitError,
    ValidationError,
    NotFoundError,
)


class ClassifAI:
    """Client for the ClassifAI API.

    A simple client for classifying text and images using the ClassifAI API.

    Args:
        api_key: API key for authenticated requests. Get yours at https://classifai.dev
                 (Optional: can be omitted for anonymous access with global rate limits)
        base_url: Base URL for the API. Defaults to https://api.classifai.dev

    Example:
        >>> from classifai import ClassifAI
        >>> client = ClassifAI(api_key="your_api_key")
        >>> result = client.classify(
        ...     content=["This product is amazing!"],
        ...     labels=["positive", "negative", "neutral"]
        ... )
        >>> print(result["label"])
        positive
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.classifai.dev",
    ):
        """Initialize the ClassifAI client.

        Args:
            api_key: Optional API key for authentication
            base_url: Base URL for the API
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

        if self.api_key:
            self.session.headers["X-API-Key"] = self.api_key

        self.session.headers["Content-Type"] = "application/json"

    def _handle_response(self, response: requests.Response) -> dict:
        """Handle API response and raise appropriate exceptions.

        Args:
            response: Response from requests

        Returns:
            Parsed JSON response

        Raises:
            AuthenticationError: If authentication fails
            RateLimitError: If rate limit is exceeded
            ValidationError: If validation fails
            NotFoundError: If resource is not found
            ClassifAIError: For other errors
        """
        try:
            data = response.json()
        except ValueError:
            data = {"error": response.text}

        if response.status_code == 200:
            return data

        error_msg = data.get("error", "Unknown error")
        detail = data.get("detail")
        if detail:
            error_msg = f"{error_msg}: {detail}"

        if response.status_code == 401:
            raise AuthenticationError(error_msg, response.status_code, data)
        elif response.status_code == 429:
            raise RateLimitError(error_msg, response.status_code, data)
        elif response.status_code == 400:
            raise ValidationError(error_msg, response.status_code, data)
        elif response.status_code == 404:
            raise NotFoundError(error_msg, response.status_code, data)
        else:
            raise ClassifAIError(error_msg, response.status_code, data)

    def classify(
        self,
        content: Union[str, List[Union[str, Path]], List[Dict[str, str]]],
        labels: Optional[List[str]] = None,
        description: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> dict:
        """Classify content items into labels.

        Classify text strings, images from files/URLs, or a mix of both.
        All content is analyzed jointly to produce a single classification result.

        The method automatically detects:
        - Plain text strings
        - Local file paths (automatically reads and encodes)
        - URLs starting with http:// or https:// (downloads and encodes)
        - Pre-formatted dicts with 'type' and 'content' keys

        Args:
            content: Content to classify. Can be:
                     - A single text string
                     - A file path (e.g., "photo.jpg")
                     - A URL (e.g., "https://example.com/image.jpg")
                     - A list mixing text, file paths, and URLs
                     - A list of dicts with 'type' and 'content' keys (advanced)
            labels: List of labels for classification (2-50 labels).
                    Required if description and project_id are not provided.
            description: Description for automatic label inference (max 500 chars).
                        Alternative to providing explicit labels.
            project_id: Project ID to use existing labels or track this classification.

        Returns:
            Dictionary with classification results:
                - labels: Dict of label scores (0.0-1.0)
                - label: Top predicted label
                - detection_id: ID for submitting feedback
                - project_id: Project this belongs to
                - labels_used: Labels that were used
                - ground_truth_url: URL for submitting feedback
                - model_used: Model identifier
                - processing_time_ms: Processing time

        Raises:
            ValidationError: If validation fails
            RateLimitError: If rate limit exceeded
            ClassifAIError: For other errors

        Examples:
            Simple text classification:
            >>> result = client.classify(
            ...     content="This is spam",
            ...     labels=["spam", "not_spam"]
            ... )

            Multiple text items:
            >>> result = client.classify(
            ...     content=["Great product!", "Fast shipping"],
            ...     labels=["positive", "negative"]
            ... )

            Image from file:
            >>> result = client.classify(
            ...     content="photo.jpg",
            ...     labels=["cat", "dog", "bird"]
            ... )

            Image from URL:
            >>> result = client.classify(
            ...     content="https://example.com/image.jpg",
            ...     labels=["cat", "dog", "bird"]
            ... )

            Mixed text and image (analyzed jointly):
            >>> result = client.classify(
            ...     content=[
            ...         "Customer reported error in checkout",
            ...         "screenshot.jpg"
            ...     ],
            ...     labels=["bug_report", "feature_request", "question"]
            ... )

            Multiple images with text:
            >>> result = client.classify(
            ...     content=[
            ...         "Before:",
            ...         "before.jpg",
            ...         "After:",
            ...         "https://example.com/after.jpg"
            ...     ],
            ...     labels=["improvement", "no_change", "worse"]
            ... )

            With automatic label inference:
            >>> result = client.classify(
            ...     content="The food was terrible",
            ...     description="Restaurant reviews"
            ... )

            Advanced - Manual format (if you need to classify literal text like "photo.jpg"):
            >>> result = client.classify(
            ...     content=[{"type": "text", "content": "photo.jpg"}],
            ...     labels=["filename", "other"]
            ... )
        """
        # Normalize content to list of dicts
        content_items = self._normalize_content(content)

        # Build request
        request_data = {"content": content_items}

        if labels:
            request_data["labels"] = labels
        if description:
            request_data["description"] = description
        if project_id:
            request_data["project_id"] = project_id

        # Make request
        response = self.session.post(
            f"{self.base_url}/classify",
            json=request_data,
        )

        return self._handle_response(response)

    def submit_feedback(
        self,
        detection_id: str,
        ground_truth: Union[str, List[str]],
    ) -> dict:
        """Submit ground truth feedback for a classification.

        Provide the correct label(s) to improve future classifications.
        New labels are automatically added to the project's label set.

        Args:
            detection_id: Detection ID from a previous classification
            ground_truth: Correct label(s). Can be a string or list of strings.

        Returns:
            Dictionary with feedback results:
                - success: Whether feedback was recorded
                - message: Status message
                - detection_id: Detection that was updated
                - updated_content_count: Number of content items
                - new_labels_added: New labels added to project

        Example:
            >>> result = client.classify(
            ...     content="This is spam",
            ...     labels=["spam", "not_spam"]
            ... )
            >>> feedback = client.submit_feedback(
            ...     result["detection_id"],
            ...     "spam"
            ... )
            >>> print(feedback["success"])
            True
        """
        # Normalize to list
        if isinstance(ground_truth, str):
            request_data = {"ground_truth": ground_truth}
        else:
            request_data = {"ground_truth_labels": ground_truth}

        response = self.session.post(
            f"{self.base_url}/ground_truth/{detection_id}",
            json=request_data,
        )

        return self._handle_response(response)

    def get_project_stats(self, project_id: str) -> dict:
        """Get statistics for a project.

        Retrieve analytics and accuracy metrics based on ground truth feedback.
        Requires ownership of the project (same API key or IP).

        Args:
            project_id: Project ID to get stats for

        Returns:
            Dictionary with project statistics:
                - project_id: Project identifier
                - name: Project name (if set)
                - total_classifications: Number of classification requests
                - total_content_items_classified: Total content items across all requests
                - total_feedback_received: Number of feedback submissions
                - accuracy_rate: Accuracy based on feedback (0.0-1.0)
                - label_distribution: Count of predicted labels
                - ground_truth_distribution: Count of ground truth labels
                - content_types_used: Content types used (text/image)
                - labels: Current project labels
                - inferred_labels: Whether labels were inferred
                - description: Project description (if any)
                - created_at: Project creation time
                - last_used_at: Last classification time

        Example:
            >>> stats = client.get_project_stats("proj_abc123")
            >>> print(f"Accuracy: {stats['accuracy_rate']:.1%}")
            Accuracy: 89.5%
        """
        response = self.session.get(
            f"{self.base_url}/projects/{project_id}/stats"
        )

        return self._handle_response(response)

    def _normalize_content(
        self, content: Union[str, List[Union[str, Path]], List[Dict[str, str]]]
    ) -> List[Dict[str, str]]:
        """Normalize content to list of content item dictionaries.

        Automatically detects and handles:
        - Plain text strings
        - File paths (reads and base64 encodes)
        - URLs (downloads and base64 encodes)
        - Already formatted dicts

        Args:
            content: Content in various formats

        Returns:
            List of dicts with 'type' and 'content' keys
        """
        # Single string - could be text, file path, or URL
        if isinstance(content, str):
            return self._process_content_items([content])

        # List of items - could be mixed types
        if isinstance(content, list):
            # Already in correct dict format
            if all(isinstance(item, dict) for item in content):
                return content

            # Process list of strings/paths/URLs
            return self._process_content_items(content)

        raise ValidationError("Content must be a string, list of strings/paths/URLs, or list of dicts")

    def _process_content_items(self, items: List[Union[str, Path]]) -> List[Dict[str, str]]:
        """Process a list of content items, detecting files and URLs.

        Args:
            items: List of strings, file paths, or URLs

        Returns:
            List of formatted content item dicts
        """
        content_items = []

        for item in items:
            item_str = str(item)

            # Check if it's a URL
            if item_str.startswith(("http://", "https://")):
                # Download image from URL
                response = requests.get(item_str, timeout=30)
                response.raise_for_status()
                image_data = response.content
                image_b64 = base64.b64encode(image_data).decode("utf-8")
                content_items.append({"type": "image", "content": image_b64})

            # Check if it's a file path
            elif isinstance(item, (str, Path)):
                path = Path(item)
                if path.exists() and path.is_file():
                    # Read and encode as image
                    with open(path, "rb") as f:
                        image_data = f.read()
                    image_b64 = base64.b64encode(image_data).decode("utf-8")
                    content_items.append({"type": "image", "content": image_b64})
                else:
                    # Treat as text string
                    content_items.append({"type": "text", "content": item_str})
            else:
                # Treat as text
                content_items.append({"type": "text", "content": item_str})

        return content_items

    def health_check(self) -> dict:
        """Check API health status.

        Returns:
            Dictionary with health status:
                - status: "healthy"
                - timestamp: Current timestamp
                - version: API version

        Example:
            >>> health = client.health_check()
            >>> print(health["status"])
            healthy
        """
        response = self.session.get(f"{self.base_url}/health")
        return self._handle_response(response)
