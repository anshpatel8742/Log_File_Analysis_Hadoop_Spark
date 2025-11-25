import streamlit as st
import pandas as pd
import os
import json
import re
import joblib

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# =========================
# PATHS
# =========================
BASE_DIR = "/home/hadoop/log_project_v2"
HDFS_JSON_PATH = "/project/logs/enhanced_logs.json"

MODEL_PATH = os.path.join(BASE_DIR, "ml", "log_classifier_v2.joblib")
VECTORIZER_PATH = os.path.join(BASE_DIR, "ml", "log_vectorizer_v2.joblib")

# =========================
# Spark + ML
# =========================
@st.cache_resource
def get_spark():
    return SparkSession.builder.appName("LogAnalyticsV2").getOrCreate()

@st.cache_resource
def load_ml():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer

def clean_text(text):
    text = re.sub(r"[|={}\[\]:,\"\']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()

# =========================
# UI
# =========================
st.set_page_config(page_title="Log Analytics Final", layout="wide")
st.title("🚀 Log Analytics Platform — Spark + ML + HDFS")

spark = get_spark()
model, vectorizer = load_ml()

# =========================
# Load data from HDFS
# =========================
try:
    df = spark.read.json(f"hdfs://{HDFS_JSON_PATH}")
except:
    st.error("❌ Could not read data from HDFS")
    st.stop()

total_logs = df.count()
st.metric("Total Logs in HDFS", f"{total_logs:,}")

if total_logs == 0:
    st.error("No logs found in HDFS")
    st.stop()

# ======================================================
# 🔹 SPARK ANALYTICS
# ======================================================
st.header("📊 Big Data Analytics using Apache Spark")

# 1. Error frequency and type
st.subheader("1. Error Frequency and Type")
error_df = (
    df.filter(
        F.col("status_code").startswith("4") |
        F.col("status_code").startswith("5")
    )
    .groupBy("status_code")
    .count()
    .orderBy("count", ascending=False)
    .toPandas()
)

if not error_df.empty:
    st.bar_chart(error_df.set_index("status_code"))
    st.dataframe(error_df)
else:
    st.info("No errors detected")

# 2. Traffic peaks by time
if "timestamp" in df.columns:
    st.subheader("2. Traffic Peaks by Hour")

    time_df = (
        df.withColumn("hour", F.hour("timestamp"))
        .groupBy("hour")
        .count()
        .orderBy("hour")
        .toPandas()
    )

    st.line_chart(time_df.set_index("hour"))
    st.dataframe(time_df)

# 3. Top IP addresses
st.subheader("3. Top 10 IP Addresses")
ip_df = (
    df.groupBy("ip_address")
    .count()
    .orderBy("count", ascending=False)
    .limit(10)
    .toPandas()
)
st.table(ip_df)

# 4. Most accessed APIs
st.subheader("4. Most Accessed APIs")
api_df = (
    df.groupBy("endpoint")
    .count()
    .orderBy("count", ascending=False)
    .limit(10)
    .toPandas()
)
st.table(api_df)

# 5. Request type distribution
st.subheader("5. Request Method Distribution")
method_df = (
    df.groupBy("method")
    .count()
    .orderBy("count", ascending=False)
    .toPandas()
)
st.bar_chart(method_df.set_index("method"))
st.dataframe(method_df)

# 6. Device / browser analytics
if "device" in df.columns:
    st.subheader("6. Device / Browser Analytics")

    device_df = (
        df.groupBy("device")
        .count()
        .orderBy("count", ascending=False)
        .toPandas()
    )

    st.bar_chart(device_df.set_index("device"))
    st.dataframe(device_df)

# 7. Server performance (response time)
if "response_time" in df.columns:
    st.subheader("7. Server Performance (Response Time in ms)")

    perf = df.select(
        F.avg("response_time").alias("avg"),
        F.max("response_time").alias("max"),
        F.min("response_time").alias("min")
    ).toPandas()

    c1, c2, c3 = st.columns(3)
    c1.metric("Average", round(perf["avg"][0], 2))
    c2.metric("Max", round(perf["max"][0], 2))
    c3.metric("Min", round(perf["min"][0], 2))

# 8. Referrer source analysis
if "referrer" in df.columns:
    st.subheader("8. Referrer Source Analysis")

    ref_df = (
        df.groupBy("referrer")
        .count()
        .orderBy("count", ascending=False)
        .limit(10)
        .toPandas()
    )

    st.table(ref_df)

# 9. Suspicious patterns (from logs)
if "suspicious_pattern" in df.columns:
    st.subheader("9. Suspicious Pattern Distribution (Metadata)")

    pattern_df = (
        df.groupBy("suspicious_pattern")
        .count()
        .orderBy("count", ascending=False)
        .toPandas()
    )

    st.bar_chart(pattern_df.set_index("suspicious_pattern"))
    st.dataframe(pattern_df)

# 10. Hadoop + Spark Scalability
st.subheader("10. Big Data Scalability (Hadoop + Spark)")
st.info("""
Data is stored on **HDFS** and processed using **Apache Spark**.
This system can scale to millions of logs by adding more nodes.
""")


# ======================================================
# 🔹 MACHINE LEARNING SECTION
# ======================================================
st.header("🤖 ML Model — Suspicious Activity Detection")

sample_size = st.slider(
    "Select sample size for ML prediction",
    min_value=50,
    max_value=2000,
    value=200,
    step=50
)

fraction = min(1.0, sample_size / total_logs)
sample_df = df.sample(fraction=fraction).limit(sample_size).toPandas()

if sample_df.empty:
    st.error("Sample returned 0 rows. Increase the sample size.")
    st.stop()

raw_texts = []
true_labels = []

for _, row in sample_df.iterrows():
    record = row.to_dict()
    label = int(record.get("label", 0))
    true_labels.append(label)

    record.pop("label", None)
    raw_texts.append(json.dumps(record))

cleaned = [clean_text(text) for text in raw_texts]

X_test = vectorizer.transform(cleaned)
preds = model.predict(X_test)

suspicious_pred = int(preds.sum())
normal_pred = int(len(preds) - suspicious_pred)
true_suspicious = int(sum(true_labels))

col1, col2, col3, col4 = st.columns(4)
col1.metric("Sample Size", len(preds))
col2.metric("Predicted Suspicious", suspicious_pred)
col3.metric("Predicted Normal", normal_pred)
col4.metric("True Suspicious", true_suspicious)

pred_df = pd.DataFrame({
    "Class": ["NORMAL", "SUSPICIOUS"],
    "Count": [normal_pred, suspicious_pred]
}).set_index("Class")

st.subheader("ML Prediction Distribution")
st.bar_chart(pred_df)

sample_df["true_label"] = ["SUSPICIOUS" if x == 1 else "NORMAL" for x in true_labels]
sample_df["predicted_label"] = ["SUSPICIOUS" if x == 1 else "NORMAL" for x in preds]

display_cols = [
    "timestamp", "ip_address", "method",
    "endpoint", "status_code",
    "true_label", "predicted_label"
]
display_cols = [c for c in display_cols if c in sample_df.columns]

st.subheader("ML Prediction Table (Sample View)")
st.dataframe(sample_df[display_cols])

# ======================================================
# FOOTER
# ======================================================
st.markdown("""
---
✅ This dashboard proves:

1. Error frequency and type  
2. Traffic peaks by date/time  
3. Top IPs  
4. Most accessed APIs  
5. Request type distribution  
6. Device / browser analytics  
7. Server performance  
8. Referrer analysis  
9. Suspicious activity detection  
10. Scalability with Hadoop + Spark  
11. ML-based classification results  

**Final Production-Ready Academic System ✅**
""")

