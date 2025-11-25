# /home/hadoop/log_project_v2/ml/train_log_classifier_v2.py

import os
import json
import re
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, "data", "enhanced_logs.json")
MODEL_FILE = os.path.join(BASE_DIR, "ml", "log_classifier_v2.joblib")
VECTORIZER_FILE = os.path.join(BASE_DIR, "ml", "log_vectorizer_v2.joblib")


def load_logs(path):
    texts = []
    labels = []
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            label = int(record.get("label", 0))

            # remove label from text features
            record_for_text = dict(record)
            record_for_text.pop("label", None)

            raw_text = json.dumps(record_for_text)
            texts.append(raw_text)
            labels.append(label)
    return texts, labels


def clean_text(text: str) -> str:
    # Very simple cleaning; you can extend
    text = re.sub(r"[\|\=\{\}\[\]\:\,\"]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()


if __name__ == "__main__":
    print(f"Loading data from {DATA_FILE} ...")
    texts, labels = load_logs(DATA_FILE)
    print("Total logs loaded:", len(texts))

    cleaned = [clean_text(t) for t in texts]

    vectorizer = TfidfVectorizer(
        max_features=4000,
        ngram_range=(1, 2),
        stop_words="english"
    )

    X = vectorizer.fit_transform(cleaned)
    y = labels

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training model...")
    model = LogisticRegression(max_iter=600)
    model.fit(X_train, y_train)
    print("Model training completed.")

    y_pred = model.predict(X_test)
    print("\n===== CLASSIFICATION REPORT =====\n")
    print(classification_report(y_test, y_pred))

    # Save artifacts
    joblib.dump(model, MODEL_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)

    print(f"✅ Saved model to {MODEL_FILE}")
    print(f"✅ Saved vectorizer to {VECTORIZER_FILE}")
