"""Tests for ClassifAI client."""

import pytest
from unittest.mock import Mock, patch, mock_open
from pathlib import Path

from classifai import ClassifAI
from classifai.exceptions import (
    AuthenticationError,
    RateLimitError,
    ValidationError,
    NotFoundError,
)


class TestClassifAIInit:
    """Test ClassifAI initialization."""

    def test_init_without_api_key(self):
        """Test initialization without API key."""
        client = ClassifAI()
        assert client.api_key is None
        assert client.base_url == "https://api.classifai.dev"
        assert "X-API-Key" not in client.session.headers

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        client = ClassifAI(api_key="test_key")
        assert client.api_key == "test_key"
        assert client.session.headers["X-API-Key"] == "test_key"

    def test_init_with_custom_base_url(self):
        """Test initialization with custom base URL."""
        client = ClassifAI(base_url="http://localhost:8000")
        assert client.base_url == "http://localhost:8000"


class TestClassify:
    """Test classify method."""

    @patch("requests.Session.post")
    def test_classify_simple_text(self, mock_post):
        """Test classifying simple text."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "label": "positive",
            "labels": {"positive": 0.9, "negative": 0.1},
            "detection_id": "det_123",
            "project_id": "proj_456",
            "labels_used": ["positive", "negative"],
            "ground_truth_url": "https://api.classifai.dev/ground_truth/det_123",
            "model_used": "test_model",
            "processing_time_ms": 100,
        }
        mock_post.return_value = mock_response

        client = ClassifAI()
        result = client.classify(
            content="This is great!",
            labels=["positive", "negative"]
        )

        assert result["label"] == "positive"
        assert result["labels"]["positive"] == 0.9
        assert result["detection_id"] == "det_123"

        # Verify request was made correctly
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0].endswith("/classify")
        assert call_args[1]["json"]["content"][0]["type"] == "text"
        assert call_args[1]["json"]["content"][0]["content"] == "This is great!"

    @patch("requests.Session.post")
    def test_classify_multiple_texts(self, mock_post):
        """Test classifying multiple text items."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "label": "positive",
            "labels": {"positive": 0.95, "negative": 0.05},
            "detection_id": "det_123",
            "project_id": "proj_456",
            "labels_used": ["positive", "negative"],
            "ground_truth_url": "https://api.classifai.dev/ground_truth/det_123",
            "model_used": "test_model",
            "processing_time_ms": 150,
        }
        mock_post.return_value = mock_response

        client = ClassifAI()
        result = client.classify(
            content=["Great product!", "Fast shipping"],
            labels=["positive", "negative"]
        )

        assert result["label"] == "positive"
        call_args = mock_post.call_args
        assert len(call_args[1]["json"]["content"]) == 2

    @patch("requests.Session.post")
    def test_classify_with_description(self, mock_post):
        """Test classify with automatic label inference."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "label": "negative",
            "labels": {"positive": 0.1, "negative": 0.9},
            "detection_id": "det_123",
            "project_id": "proj_456",
            "labels_used": ["positive", "negative", "neutral"],
            "ground_truth_url": "https://api.classifai.dev/ground_truth/det_123",
            "model_used": "test_model",
            "processing_time_ms": 200,
        }
        mock_post.return_value = mock_response

        client = ClassifAI()
        result = client.classify(
            content="The food was terrible",
            description="Restaurant reviews"
        )

        assert result["label"] == "negative"
        call_args = mock_post.call_args
        assert call_args[1]["json"]["description"] == "Restaurant reviews"

    @patch("requests.Session.post")
    def test_classify_with_project_id(self, mock_post):
        """Test classify with project ID."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "label": "positive",
            "labels": {"positive": 0.9, "negative": 0.1},
            "detection_id": "det_123",
            "project_id": "my-project",
            "labels_used": ["positive", "negative"],
            "ground_truth_url": "https://api.classifai.dev/ground_truth/det_123",
            "model_used": "test_model",
            "processing_time_ms": 100,
        }
        mock_post.return_value = mock_response

        client = ClassifAI()
        result = client.classify(
            content="Great!",
            project_id="my-project"
        )

        assert result["project_id"] == "my-project"
        call_args = mock_post.call_args
        assert call_args[1]["json"]["project_id"] == "my-project"


class TestClassifyFiles:
    """Test classify method with files and URLs."""

    @patch("requests.Session.post")
    @patch("builtins.open", new_callable=mock_open, read_data=b"fake_image_data")
    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.is_file", return_value=True)
    def test_classify_from_local_file(self, mock_is_file, mock_exists, mock_file, mock_post):
        """Test classifying from local image file."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "label": "cat",
            "labels": {"cat": 0.9, "dog": 0.1},
            "detection_id": "det_123",
            "project_id": "proj_456",
            "labels_used": ["cat", "dog"],
            "ground_truth_url": "https://api.classifai.dev/ground_truth/det_123",
            "model_used": "test_model",
            "processing_time_ms": 300,
        }
        mock_post.return_value = mock_response

        client = ClassifAI()
        result = client.classify(
            content=["photo.jpg"],
            labels=["cat", "dog"]
        )

        assert result["label"] == "cat"
        call_args = mock_post.call_args
        assert call_args[1]["json"]["content"][0]["type"] == "image"

    @patch("requests.Session.post")
    @patch("requests.get")
    def test_classify_from_url(self, mock_get, mock_post):
        """Test classifying image from URL."""
        # Mock URL download
        mock_url_response = Mock()
        mock_url_response.content = b"fake_image_data"
        mock_url_response.raise_for_status = Mock()
        mock_get.return_value = mock_url_response

        # Mock classify response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "label": "cat",
            "labels": {"cat": 0.8, "dog": 0.2},
            "detection_id": "det_123",
            "project_id": "proj_456",
            "labels_used": ["cat", "dog"],
            "ground_truth_url": "https://api.classifai.dev/ground_truth/det_123",
            "model_used": "test_model",
            "processing_time_ms": 350,
        }
        mock_post.return_value = mock_response

        client = ClassifAI()
        result = client.classify(
            content=["https://example.com/photo.jpg"],
            labels=["cat", "dog"]
        )

        assert result["label"] == "cat"
        mock_get.assert_called_once_with("https://example.com/photo.jpg", timeout=30)

    @patch("requests.Session.post")
    def test_classify_non_existent_file_as_text(self, mock_post):
        """Test that non-existent file paths are treated as text."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "label": "positive",
            "labels": {"positive": 0.9, "negative": 0.1},
            "detection_id": "det_123",
            "project_id": "proj_456",
            "labels_used": ["positive", "negative"],
            "ground_truth_url": "https://api.classifai.dev/ground_truth/det_123",
            "model_used": "test_model",
            "processing_time_ms": 100,
        }
        mock_post.return_value = mock_response

        client = ClassifAI()
        result = client.classify(
            content=["This is text not a file"],
            labels=["positive", "negative"]
        )

        assert result["label"] == "positive"
        call_args = mock_post.call_args
        assert call_args[1]["json"]["content"][0]["type"] == "text"


class TestSubmitFeedback:
    """Test submit_feedback method."""

    @patch("requests.Session.post")
    def test_submit_feedback_single_label(self, mock_post):
        """Test submitting single ground truth label."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "message": "Feedback recorded",
            "detection_id": "det_123",
            "updated_content_count": 1,
            "new_labels_added": [],
        }
        mock_post.return_value = mock_response

        client = ClassifAI()
        result = client.submit_feedback("det_123", "positive")

        assert result["success"] is True
        assert result["detection_id"] == "det_123"
        call_args = mock_post.call_args
        assert call_args[0][0].endswith("/ground_truth/det_123")
        assert call_args[1]["json"]["ground_truth"] == "positive"

    @patch("requests.Session.post")
    def test_submit_feedback_multiple_labels(self, mock_post):
        """Test submitting multiple ground truth labels."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "message": "Feedback recorded",
            "detection_id": "det_123",
            "updated_content_count": 1,
            "new_labels_added": ["helpful"],
        }
        mock_post.return_value = mock_response

        client = ClassifAI()
        result = client.submit_feedback("det_123", ["positive", "helpful"])

        assert result["success"] is True
        assert "helpful" in result["new_labels_added"]
        call_args = mock_post.call_args
        assert call_args[1]["json"]["ground_truth_labels"] == ["positive", "helpful"]


class TestGetProjectStats:
    """Test get_project_stats method."""

    @patch("requests.Session.get")
    def test_get_project_stats(self, mock_get):
        """Test getting project statistics."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "project_id": "proj_123",
            "name": "Test Project",
            "total_classifications": 100,
            "total_content_items_classified": 250,
            "total_feedback_received": 50,
            "accuracy_rate": 0.85,
            "label_distribution": {"positive": 60, "negative": 40},
            "ground_truth_distribution": {"positive": 30, "negative": 20},
            "content_types_used": ["text"],
            "labels": ["positive", "negative"],
            "inferred_labels": False,
            "description": None,
            "created_at": "2025-01-01T00:00:00",
            "last_used_at": "2025-01-02T00:00:00",
        }
        mock_get.return_value = mock_response

        client = ClassifAI()
        stats = client.get_project_stats("proj_123")

        assert stats["project_id"] == "proj_123"
        assert stats["total_classifications"] == 100
        assert stats["accuracy_rate"] == 0.85
        mock_get.assert_called_once_with(
            "https://api.classifai.dev/projects/proj_123/stats"
        )


class TestErrorHandling:
    """Test error handling."""

    @patch("requests.Session.post")
    def test_rate_limit_error(self, mock_post):
        """Test rate limit error handling."""
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.json.return_value = {
            "error": "Rate limit exceeded",
            "detail": "10 per minute",
        }
        mock_post.return_value = mock_response

        client = ClassifAI()
        with pytest.raises(RateLimitError) as exc_info:
            client.classify(content="test", labels=["a", "b"])

        assert "Rate limit exceeded" in str(exc_info.value)
        assert exc_info.value.status_code == 429

    @patch("requests.Session.post")
    def test_authentication_error(self, mock_post):
        """Test authentication error handling."""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.json.return_value = {
            "error": "Invalid API key",
        }
        mock_post.return_value = mock_response

        client = ClassifAI(api_key="invalid_key")
        with pytest.raises(AuthenticationError) as exc_info:
            client.classify(content="test", labels=["a", "b"])

        assert exc_info.value.status_code == 401

    @patch("requests.Session.post")
    def test_validation_error(self, mock_post):
        """Test validation error handling."""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.json.return_value = {
            "error": "Validation error",
            "detail": "Invalid labels",
        }
        mock_post.return_value = mock_response

        client = ClassifAI()
        with pytest.raises(ValidationError) as exc_info:
            client.classify(content="test", labels=["a"])

        assert exc_info.value.status_code == 400

    @patch("requests.Session.get")
    def test_not_found_error(self, mock_get):
        """Test not found error handling."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.json.return_value = {
            "error": "Project not found",
        }
        mock_get.return_value = mock_response

        client = ClassifAI()
        with pytest.raises(NotFoundError) as exc_info:
            client.get_project_stats("invalid_project")

        assert exc_info.value.status_code == 404


class TestHealthCheck:
    """Test health_check method."""

    @patch("requests.Session.get")
    def test_health_check(self, mock_get):
        """Test health check."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "healthy",
            "timestamp": 1234567890,
            "version": "1.0.0",
        }
        mock_get.return_value = mock_response

        client = ClassifAI()
        health = client.health_check()

        assert health["status"] == "healthy"
        mock_get.assert_called_once_with("https://api.classifai.dev/health")
