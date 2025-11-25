import os
import json
import re
import joblib
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import pandas as pd

# -----------------------
# PATHS
# -----------------------
BASE_DIR = "/home/hadoop/log_project_v2"
HDFS_JSON_PATH = "/project/logs/enhanced_logs.json"

MODEL_PATH = os.path.join(BASE_DIR, "ml", "log_classifier_v2.joblib")
VECTORIZER_PATH = os.path.join(BASE_DIR, "ml", "log_vectorizer_v2.joblib")

SAMPLE_SIZE = 1000   # change this if needed

# -----------------------
# Spark session
# -----------------------
spark = SparkSession.builder.appName("ML-Test-Only").getOrCreate()

print("\n✅ Spark started")

# -----------------------
# Load data from HDFS
# -----------------------
df = spark.read.json(f"hdfs://{HDFS_JSON_PATH}")

total_count = df.count()
print(f"\n✅ Total logs in HDFS: {total_count}")

if total_count == 0:
    print("\n❌ No data found in HDFS file")
    exit()

# -----------------------
# Smart sampling
# -----------------------
fraction = min(1.0, SAMPLE_SIZE / total_count)
sample_df = df.sample(fraction=fraction).limit(SAMPLE_SIZE).toPandas()

if sample_df.empty:
    print("\n❌ Sample returned 0 rows. Increase SAMPLE_SIZE.")
    exit()

print(f"✅ Sample size: {len(sample_df)} logs")

# -----------------------
# Load model + vectorizer
# -----------------------
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

print("\n✅ Model and vectorizer loaded")

# -----------------------
# Clean + prepare text
# -----------------------
def clean_text(text: str) -> str:
    text = re.sub(r"[|={}\[\]:,\"\']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()

raw_texts = []
true_labels = []

for _, row in sample_df.iterrows():
    record = row.to_dict()

    label = int(record.get("label", 0))
    true_labels.append(label)

    record.pop("label", None)
    raw_texts.append(json.dumps(record))

cleaned = [clean_text(text) for text in raw_texts]

# -----------------------
# ML Prediction
# -----------------------
X_test = vectorizer.transform(cleaned)
preds = model.predict(X_test)

suspicious_pred = int(preds.sum())
normal_pred = int(len(preds) - suspicious_pred)
true_suspicious = int(sum(true_labels))

# -----------------------
# RESULTS
# -----------------------
print("\n========== ML TEST RESULTS ==========")
print(f"Total Sample Logs      : {len(preds)}")
print(f"Predicted Suspicious   : {suspicious_pred}")
print(f"Predicted Normal       : {normal_pred}")
print(f"True Suspicious (label): {true_suspicious}")
print("=====================================\n")

# Show sample table
sample_df["predicted_label"] = ["SUSPICIOUS" if x == 1 else "NORMAL" for x in preds]
sample_df["true_label"] = ["SUSPICIOUS" if x == 1 else "NORMAL" for x in true_labels]

print("Sample Predictions:\n")
print(sample_df[[
    "timestamp",
    "ip_address",
    "method",
    "endpoint",
    "status_code",
    "true_label",
    "predicted_label"
]].head(20))

print("\n✅ ML Test Completed Successfully")
