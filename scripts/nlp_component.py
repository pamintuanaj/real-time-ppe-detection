import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Lightweight text classifier for safety messages

DATA = [
    ("worker wearing hardhat and vest", "compliant"),
    ("employee has helmet and safety vest", "compliant"),
    ("person wearing mask hardhat and vest", "compliant"),
    ("worker missing hardhat", "non_compliant"),
    ("person without safety vest", "non_compliant"),
    ("employee not wearing mask", "non_compliant"),
    ("construction worker fully equipped", "compliant"),
    ("worker has no helmet", "non_compliant"),
    ("worker wearing all protective equipment", "compliant"),
    ("person missing ppe", "non_compliant"),
    ("mask and hardhat detected", "compliant"),
    ("no vest detected", "non_compliant"),
]

def train_text_classifier():
    texts = [x[0] for x in DATA]
    labels = [x[1] for x in DATA]

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=labels
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True, zero_division=0)

    return model, acc, report

def generate_safety_message(detected_items):
    if not detected_items:
        return "No PPE items detected."

    items = ", ".join(detected_items)
    return f"Detected PPE items: {items}."

def classify_message(model, text):
    return model.predict([text])[0]

def main():
    model, acc, report = train_text_classifier()

    output_dir = Path("models/nlp")
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "accuracy": float(acc),
        "report": report
    }

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    sample = generate_safety_message(["hardhat", "vest"])
    pred = classify_message(model, sample.lower())

    print("NLP prototype complete.")
    print(f"Accuracy: {acc:.4f}")
    print("Sample message:", sample)
    print("Predicted class:", pred)
    print(f"Saved metrics to: {output_dir / 'metrics.json'}")

if __name__ == "__main__":
    main()