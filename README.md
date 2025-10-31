# ClassifAI Python Client

A dead simple Python client for [classifai.dev](https://classifai.dev) - a self-improving, multimodal classification API for developers and AI agents.

## Features

- Classify text, images, or both together
- Automatic label inference from descriptions
- Submit ground truth feedback to improve accuracy
- Track project statistics and accuracy
- Support for local files and image URLs
- Simple, intuitive API

## Installation

```bash
pip install classifai
```

## Quick Start

```python
from classifai import ClassifAI

# Initialize client
client = ClassifAI(api_key="your_api_key")

# Classify text
result = client.classify(
    content="This product is amazing!",
    labels=["positive", "negative", "neutral"]
)

print(f"Label: {result['label']}")  # positive
print(f"Confidence: {result['labels']['positive']:.2%}")  # 92%
```

## Usage Examples

### Text Classification

```python
# Simple text classification
result = client.classify(
    content="This is spam",
    labels=["spam", "not_spam"]
)

# Multiple text items (analyzed jointly)
result = client.classify(
    content=["Great product!", "Fast shipping", "Highly recommend"],
    labels=["positive", "negative", "neutral"]
)

# Automatic label inference
result = client.classify(
    content="The food was terrible and service was slow",
    description="Restaurant reviews"
)
# API automatically infers labels like: ["positive", "negative", "neutral", "mixed"]
```

### Image Classification

```python
# Classify from local file
result = client.classify(
    content="photo.jpg",
    labels=["cat", "dog", "bird", "other"]
)

# Classify from URL
result = client.classify(
    content="https://example.com/image.jpg",
    labels=["cat", "dog", "bird", "other"]
)
```

### Mixed Text and Images

The real power of ClassifAI is analyzing text and images together:

```python
# Support ticket routing with screenshot
result = client.classify(
    content=[
        "Customer reported: Cannot complete checkout",
        "screenshot.jpg"
    ],
    labels=["bug_report", "feature_request", "question", "billing_issue"]
)

# Product review with multiple images
result = client.classify(
    content=[
        "Before treatment:",
        "before.jpg",
        "After 30 days:",
        "after.jpg",
        "Amazing results!"
    ],
    labels=["positive", "negative", "neutral"]
)

# Social media moderation
result = client.classify(
    content=[
        "Check out this offer!",
        "https://example.com/promo.jpg"
    ],
    labels=["spam", "legitimate", "suspicious"]
)
```

### Ground Truth Feedback

Improve accuracy by providing correct labels:

```python
# Make a classification
result = client.classify(
    content="This is spam",
    labels=["spam", "not_spam"]
)

# Submit feedback if prediction was wrong
feedback = client.submit_feedback(
    detection_id=result["detection_id"],
    ground_truth="spam"
)

print(feedback["success"])  # True
```

### Project Statistics

Track accuracy and performance:

```python
stats = client.get_project_stats(result["project_id"])

print(f"Total classifications: {stats['total_classifications']}")
print(f"Accuracy: {stats['accuracy_rate']:.1%}")
print(f"Label distribution: {stats['label_distribution']}")
```

### Using Project IDs

Reuse label sets across requests:

```python
# First request creates project
result1 = client.classify(
    content="This is great!",
    labels=["positive", "negative"],
    project_id="sentiment-v1"
)

# Subsequent requests reuse the same labels
result2 = client.classify(
    content="This is terrible!",
    project_id="sentiment-v1"  # Uses ["positive", "negative"] from above
)
```

## Advanced Usage

### Manual Content Format

For full control, use the dict format directly:

```python
import base64

# Read and encode image manually
with open("image.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

# Classify with explicit content items
result = client.classify(
    content=[
        {"type": "text", "content": "What is this?"},
        {"type": "image", "content": image_b64}
    ],
    labels=["cat", "dog", "bird"]
)
```

### Error Handling

```python
from classifai import ClassifAI, RateLimitError, ValidationError

client = ClassifAI(api_key="your_key")

try:
    result = client.classify(
        content="Test",
        labels=["label1", "label2"]
    )
except RateLimitError as e:
    print(f"Rate limit exceeded: {e}")
except ValidationError as e:
    print(f"Invalid request: {e}")
```

## API Reference

### ClassifAI(api_key, base_url="https://api.classifai.dev")

Initialize the client.

**Parameters:**
- `api_key` (str): API key for authentication. Get yours at [classifai.dev](https://classifai.dev). (Optional: can be omitted for anonymous access with global rate limits)
- `base_url` (str): Base URL for the API

### classify(content, labels=None, description=None, project_id=None)

Classify text, images from files/URLs, or a mix of both.

Automatically detects and handles:
- Plain text strings
- Local file paths (reads and encodes images)
- URLs starting with http:// or https:// (downloads and encodes)
- Pre-formatted dicts with 'type' and 'content' keys

**Parameters:**
- `content` (str | list[str | Path] | list[dict]): Content to classify - can be text, file paths, URLs, or a mix
- `labels` (list[str], optional): Explicit labels (2-50 labels)
- `description` (str, optional): Description for automatic label inference
- `project_id` (str, optional): Project ID for reusing labels

**Returns:** dict with `label`, `labels`, `detection_id`, `project_id`, etc.

### submit_feedback(detection_id, ground_truth)

Submit ground truth feedback.

**Parameters:**
- `detection_id` (str): Detection ID from previous classification
- `ground_truth` (str | list[str]): Correct label(s)

**Returns:** dict with `success`, `message`, `new_labels_added`, etc.

### get_project_stats(project_id)

Get project statistics.

**Parameters:**
- `project_id` (str): Project ID

**Returns:** dict with `accuracy_rate`, `total_classifications`, `label_distribution`, etc.

## Rate Limits

| Tier | Rate Limits | Size Limit |
|------|-------------|------------|
| Free | 10/min, 100/day | 500 KB |
| Hobby | 10/min, 1,000/day | 2.5 MB |
| Production | 100/min, 10,000/day | 10 MB |

Get your API key at [classifai.dev](https://classifai.dev)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details.

## Links

- **Website:** https://classifai.dev
- **Documentation:** https://classifai.dev/docs
- **API Docs:** https://api.classifai.dev/docs
- **GitHub:** https://github.com/veezbo/classifai-python
