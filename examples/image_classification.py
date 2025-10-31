"""Image classification examples for ClassifAI Python client."""

from classifai import ClassifAI

# Initialize client
client = ClassifAI(api_key="your_api_key_here")


# Example 1: Classify a single image from local file
print("Example 1: Single image from file")
result = client.classify(
    content=["cat_photo.jpg"],
    labels=["cat", "dog", "bird", "other"]
)
print(f"Label: {result['label']}")
print(f"Confidence: {result['labels'][result['label']]:.2%}")
print()


# Example 2: Classify image from URL
print("Example 2: Image from URL")
result = client.classify(
    content=["https://example.com/product-image.jpg"],
    labels=["damaged", "good_condition", "new"]
)
print(f"Label: {result['label']}")
print()


# Example 3: Text with image - Support ticket routing
print("Example 3: Support ticket with screenshot")
result = client.classify(
    content=[
        "User reported: Getting error 500 when trying to checkout",
        "error_screenshot.png"
    ],
    labels=["bug_report", "feature_request", "user_error", "billing_issue"]
)
print(f"Route to: {result['label']}")
print(f"Detection ID: {result['detection_id']}")
print()


# Example 4: Multiple images with text - Product review
print("Example 4: Product review with before/after photos")
result = client.classify(
    content=[
        "Before using the product:",
        "before.jpg",
        "After 30 days:",
        "after.jpg",
        "Amazing transformation! Highly recommend."
    ],
    labels=["positive", "negative", "neutral", "mixed"]
)
print(f"Sentiment: {result['label']}")
print()


# Example 5: Social media content moderation
print("Example 5: Content moderation with image")
result = client.classify(
    content=[
        "Check out this amazing deal! Limited time only!",
        "https://example.com/promo.jpg"
    ],
    labels=["spam", "legitimate", "suspicious", "inappropriate"]
)
print(f"Classification: {result['label']}")
print()


# Example 6: Manual base64 encoding (advanced)
print("Example 6: Manual base64 encoding")
import base64

with open("photo.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

result = client.classify(
    content=[
        {"type": "text", "content": "What animal is this?"},
        {"type": "image", "content": image_data}
    ],
    labels=["cat", "dog", "bird", "other"]
)
print(f"Label: {result['label']}")
print()


# Example 7: Multiple images from different sources
print("Example 7: Mix of local files and URLs")
result = client.classify(
    content=[
        "Comparing products:",
        "local_product.jpg",
        "vs",
        "https://example.com/competitor_product.jpg"
    ],
    labels=["our_product_better", "competitor_better", "similar"]
)
print(f"Result: {result['label']}")
print()


# Example 8: Image classification with automatic label inference
print("Example 8: Image with label inference")
result = client.classify(
    content=["street_scene.jpg"],
    description="Urban street photography scenes"
)
print(f"Inferred labels: {result['labels_used']}")
print(f"Label: {result['label']}")
print()


# Example 9: Real-world use case - Receipt processing
print("Example 9: Receipt classification")
result = client.classify(
    content=[
        "Please reimburse this expense",
        "receipt.jpg"
    ],
    labels=["valid_expense", "invalid_expense", "needs_clarification"]
)
print(f"Status: {result['label']}")

# Submit feedback
feedback = client.submit_feedback(
    detection_id=result["detection_id"],
    ground_truth="valid_expense"
)
print(f"Feedback recorded: {feedback['success']}")
print()


# Example 10: Medical/Healthcare - X-ray analysis routing
print("Example 10: Medical imaging triage")
result = client.classify(
    content=[
        "Patient complaint: chest pain",
        "chest_xray.jpg"
    ],
    labels=["urgent", "routine", "normal", "needs_specialist"]
)
print(f"Priority: {result['label']}")
print()
