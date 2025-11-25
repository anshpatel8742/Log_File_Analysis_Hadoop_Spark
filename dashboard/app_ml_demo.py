import streamlit as st

# ✅ MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="ML Demo System", layout="centered")

import os
import json
import re
import subprocess
import joblib
from pyspark.sql import SparkSession
import pandas as pd

# =========================
# PATHS
# =========================
BASE_DIR = "/home/hadoop/log_project_v2"

GEN_SCRIPT = os.path.join(BASE_DIR, "data", "enhanced_log_generator_v2.py")
LOCAL_FILE = os.path.join(BASE_DIR, "data", "enhanced_logs.json")

HDFS_DEMO_PATH = "/project/logs/demo_logs.json"

MODEL_PATH = os.path.join(BASE_DIR, "ml", "log_classifier_v2.joblib")
VECTORIZER_PATH = os.path.join(BASE_DIR, "ml", "log_vectorizer_v2.joblib")

# =========================
# Spark
# =========================
@st.cache_resource
def get_spark():
    return SparkSession.builder.appName("ML-Demo").getOrCreate()

spark = get_spark()

# =========================
# Load ML
# =========================
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer

model, vectorizer = load_model()

# =========================
# Helpers
# =========================
def clean_text(text):
    text = re.sub(r"[|={}\[\]:,\"\']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()

def run_cmd(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout, result.stderr, result.returncode

# =========================
# UI
# =========================
st.title("🎯 ML Demo — Generate Logs + Run Model (HDFS)")

st.write("""
This dashboard is only for **demonstration** of:
- Data generation  
- HDFS upload  
- ML prediction (no retraining)  
""")

log_count = st.number_input(
    "Number of logs to generate",
    min_value=50,
    max_value=5000,
    value=200,
    step=50
)

# =========================
# MAIN BUTTON
# =========================
if st.button("🚀 Generate Logs + Run Model"):

    # 1. Generate logs
    with st.spinner("Generating logs..."):
        gen_cmd = f"python3 {GEN_SCRIPT} --total {log_count}"
        out, err, code = run_cmd(gen_cmd)

        if code != 0:
            st.error("❌ Log generation failed")
            st.text(err)
            st.stop()

        st.success("✅ Logs generated locally")

    # 2. Upload to HDFS
    with st.spinner("Uploading to HDFS..."):
        run_cmd("hdfs dfs -mkdir -p /project/logs")

        upload_cmd = f"hdfs dfs -put -f {LOCAL_FILE} {HDFS_DEMO_PATH}"
        out2, err2, code2 = run_cmd(upload_cmd)

        if code2 != 0:
            st.error("❌ Upload to HDFS failed")
            st.text(err2)
            st.stop()

        st.success("✅ Uploaded logs to HDFS")

    # 3. Run ML on HDFS data
    with st.spinner("Running ML model on new data..."):

        df = spark.read.json(f"hdfs://{HDFS_DEMO_PATH}")
        total = df.count()

        if total == 0:
            st.error("No logs found in HDFS")
            st.stop()

        st.success(f"✅ Loaded {total} logs from HDFS")

        sample_df = df.toPandas()

        raw_texts = []
        true_labels = []

        for _, row in sample_df.iterrows():
            record = row.to_dict()

            # Save true label
            label = int(record.get("label", 0))
            true_labels.append(label)

            # ✅ THIS IS THE CRITICAL FIX
            log_text = f"{record.get('ip_address','')} {record.get('method','')} {record.get('endpoint','')} {record.get('status_code','')} {record.get('device','')} {record.get('browser','')} {record.get('referrer','')}"

            raw_texts.append(log_text)

        cleaned = [clean_text(text) for text in raw_texts]

        # DEBUG – show input to model
        st.write("🔍 Sample input to model:", cleaned[:5])

        X_test = vectorizer.transform(cleaned)
        preds = model.predict(X_test)

        suspicious_pred = int(preds.sum())
        normal_pred = int(len(preds) - suspicious_pred)
        true_suspicious = int(sum(true_labels))

        # =========================
        # RESULTS
        # =========================
        st.header("📊 ML Prediction Result")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Logs", total)
        col2.metric("Predicted Suspicious", suspicious_pred)
        col3.metric("Predicted Normal", normal_pred)
        col4.metric("True Suspicious", true_suspicious)

        result_df = pd.DataFrame({
            "Class": ["NORMAL", "SUSPICIOUS"],
            "Count": [normal_pred, suspicious_pred]
        }).set_index("Class")

        st.bar_chart(result_df)

        # Table
        sample_df["true_label"] = ["SUSPICIOUS" if x == 1 else "NORMAL" for x in true_labels]
        sample_df["predicted_label"] = ["SUSPICIOUS" if x == 1 else "NORMAL" for x in preds]

        cols = [
            "timestamp", "ip_address", "method",
            "endpoint", "status_code",
            "true_label", "predicted_label"
        ]

        cols = [c for c in cols if c in sample_df.columns]

        st.subheader("Prediction Table")
        st.dataframe(sample_df[cols])

        st.success("✅ ML demo completed successfully")

