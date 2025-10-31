"""Basic usage examples for ClassifAI Python client."""

from classifai import ClassifAI

# Initialize client with your API key
# Get your key at: https://classifai.dev
client = ClassifAI(api_key="your_api_key_here")


# Example 1: Simple text classification
print("Example 1: Simple text classification")
result = client.classify(
    content="This product exceeded my expectations!",
    labels=["positive", "negative", "neutral"]
)
print(f"Label: {result['label']}")
print(f"Scores: {result['labels']}")
print(f"Detection ID: {result['detection_id']}")
print()


# Example 2: Multiple text items analyzed together
print("Example 2: Multiple text items")
result = client.classify(
    content=[
        "Great product!",
        "Fast shipping",
        "Highly recommend"
    ],
    labels=["positive", "negative", "neutral"]
)
print(f"Label: {result['label']}")
print(f"Confidence: {result['labels'][result['label']]:.2%}")
print()


# Example 3: Automatic label inference
print("Example 3: Automatic label inference")
result = client.classify(
    content="The pasta was cold and the waiter was rude",
    description="Restaurant reviews from customers"
)
print(f"Inferred labels: {result['labels_used']}")
print(f"Label: {result['label']}")
print()


# Example 4: Submit ground truth feedback
print("Example 4: Submit feedback")
result = client.classify(
    content="Click here to claim your prize!",
    labels=["spam", "not_spam"]
)
print(f"Predicted: {result['label']}")

# Provide correct label
feedback = client.submit_feedback(
    detection_id=result["detection_id"],
    ground_truth="spam"
)
print(f"Feedback recorded: {feedback['success']}")
print()


# Example 5: Get project statistics
print("Example 5: Project statistics")
stats = client.get_project_stats(result["project_id"])
print(f"Total classifications: {stats['total_classifications']}")
print(f"Feedback received: {stats['total_feedback_received']}")
if stats['accuracy_rate']:
    print(f"Accuracy: {stats['accuracy_rate']:.1%}")
print(f"Label distribution: {stats['label_distribution']}")
print()


# Example 6: Using project IDs
print("Example 6: Reusing project labels")
result1 = client.classify(
    content="Love it!",
    labels=["positive", "negative"],
    project_id="sentiment-analysis-v1"
)
print(f"First: {result1['label']}")

# Reuse the same labels
result2 = client.classify(
    content="Hate it!",
    project_id="sentiment-analysis-v1"
)
print(f"Second: {result2['label']}")
print(f"Same project: {result1['project_id'] == result2['project_id']}")
