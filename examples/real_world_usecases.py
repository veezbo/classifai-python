"""Real-world use case examples for ClassifAI."""

from classifai import ClassifAI, RateLimitError, ValidationError

client = ClassifAI(api_key="your_api_key_here")


# Use Case 1: Customer Support Ticket Routing
print("=" * 60)
print("Use Case 1: Customer Support Ticket Routing")
print("=" * 60)

def route_support_ticket(message, screenshot_path=None):
    """Route customer support tickets to the right team."""
    content = [message]
    if screenshot_path:
        content.append(screenshot_path)

    result = client.classify(
        content=content,
        labels=[
            "technical_bug",
            "billing_question",
            "feature_request",
            "account_access",
            "general_inquiry"
        ],
        project_id="support-routing-v1"
    )

    return result["label"], result["detection_id"]

# Example tickets
ticket1_route, ticket1_id = route_support_ticket(
    "I can't log in to my account, getting error 401",
    "login_error.png"
)
print(f"Ticket 1 → {ticket1_route}")

ticket2_route, ticket2_id = route_support_ticket(
    "When will the new dashboard feature be available?"
)
print(f"Ticket 2 → {ticket2_route}")
print()


# Use Case 2: Content Moderation for Social Media
print("=" * 60)
print("Use Case 2: Social Media Content Moderation")
print("=" * 60)

def moderate_post(text, image_urls=None):
    """Moderate user-generated content."""
    content = [text]
    if image_urls:
        content.extend(image_urls)

    result = client.classify(
        content=content,
        labels=["safe", "spam", "inappropriate", "requires_review"],
        project_id="content-moderation-v1"
    )

    return result

post1 = moderate_post(
    "Check out my new art project!",
    ["https://example.com/art.jpg"]
)
print(f"Post 1: {post1['label']} (confidence: {post1['labels'][post1['label']]:.1%})")

post2 = moderate_post(
    "BUY NOW! Limited time offer! Click here!"
)
print(f"Post 2: {post2['label']}")
print()


# Use Case 3: E-commerce Product Review Sentiment
print("=" * 60)
print("Use Case 3: Product Review Sentiment Analysis")
print("=" * 60)

def analyze_review(review_text, review_images=None):
    """Analyze product reviews with text and images."""
    content = [review_text]
    if review_images:
        content.extend(review_images)

    result = client.classify(
        content=content,
        labels=["very_positive", "positive", "neutral", "negative", "very_negative"],
        project_id="review-sentiment-v1"
    )

    return result

review1 = analyze_review(
    "This product changed my life! Here are before/after pics:",
    ["before.jpg", "after.jpg"]
)
print(f"Review 1: {review1['label']}")

# Track accuracy with feedback
stats = client.get_project_stats("review-sentiment-v1")
if stats['accuracy_rate']:
    print(f"Review system accuracy: {stats['accuracy_rate']:.1%}")
print()


# Use Case 4: Document Classification
print("=" * 60)
print("Use Case 4: Document Type Classification")
print("=" * 60)

def classify_document(doc_path):
    """Classify scanned documents."""
    result = client.classify(
        content=[doc_path],
        labels=[
            "invoice",
            "receipt",
            "contract",
            "id_document",
            "other"
        ],
        project_id="document-classifier-v1"
    )

    return result

doc1 = classify_document("scanned_doc.pdf")
print(f"Document type: {doc1['label']}")
print()


# Use Case 5: Medical Triage System
print("=" * 60)
print("Use Case 5: Medical Image Triage")
print("=" * 60)

def triage_medical_case(symptoms, images=None):
    """Triage medical cases by urgency."""
    content = [f"Patient symptoms: {symptoms}"]
    if images:
        content.extend(images)

    result = client.classify(
        content=content,
        labels=["emergency", "urgent", "routine", "non_urgent"],
        project_id="medical-triage-v1"
    )

    return result

case1 = triage_medical_case(
    "Severe chest pain, shortness of breath",
    ["ecg_reading.jpg"]
)
print(f"Triage level: {case1['label']}")
print(f"Confidence: {case1['labels'][case1['label']]:.1%}")
print()


# Use Case 6: Quality Control in Manufacturing
print("=" * 60)
print("Use Case 6: Manufacturing Quality Control")
print("=" * 60)

def inspect_product(product_images, notes=""):
    """Inspect products for defects."""
    content = []
    if notes:
        content.append(f"Inspector notes: {notes}")
    content.extend(product_images)

    result = client.classify(
        content=content,
        labels=["pass", "fail", "needs_review"],
        project_id="qc-inspection-v1"
    )

    return result

inspection = inspect_product(
    ["product_front.jpg", "product_back.jpg"],
    "Minor discoloration on back panel"
)
print(f"QC Result: {inspection['label']}")

# Provide ground truth after manual review
client.submit_feedback(
    detection_id=inspection["detection_id"],
    ground_truth="needs_review"
)
print("Feedback submitted to improve accuracy")
print()


# Use Case 7: Real Estate Listing Classification
print("=" * 60)
print("Use Case 7: Real Estate Property Classification")
print("=" * 60)

def classify_property(description, photos):
    """Classify property type from listing."""
    content = [description]
    content.extend(photos)

    result = client.classify(
        content=content,
        labels=[
            "single_family",
            "condo",
            "townhouse",
            "multi_family",
            "commercial"
        ],
        project_id="real-estate-v1"
    )

    return result

property1 = classify_property(
    "Beautiful 3BR/2BA with backyard",
    ["exterior.jpg", "kitchen.jpg", "backyard.jpg"]
)
print(f"Property type: {property1['label']}")
print()


# Use Case 8: Error Handling and Retry Logic
print("=" * 60)
print("Use Case 8: Error Handling Best Practices")
print("=" * 60)

def classify_with_retry(content, labels, max_retries=3):
    """Classify with automatic retry on rate limits."""
    import time

    for attempt in range(max_retries):
        try:
            result = client.classify(
                content=content,
                labels=labels
            )
            return result

        except RateLimitError as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Rate limited, retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise

        except ValidationError as e:
            print(f"Validation error: {e}")
            raise

try:
    result = classify_with_retry(
        content="Test message",
        labels=["category1", "category2"]
    )
    print(f"Classification successful: {result['label']}")
except Exception as e:
    print(f"Failed after retries: {e}")

print()
print("=" * 60)
print("All use cases completed!")
print("=" * 60)
