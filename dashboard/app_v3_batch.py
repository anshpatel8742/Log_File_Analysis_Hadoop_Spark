import streamlit as st
import pandas as pd
import os
import json
import subprocess
import re
from datetime import datetime

import joblib
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
LOCAL_BATCH_DIR = os.path.join(DATA_DIR, "batches")
DEFAULT_GEN_FILE = os.path.join(DATA_DIR, "enhanced_logs.json")

GEN_SCRIPT = os.path.join(DATA_DIR, "enhanced_log_generator_v2.py")
HDFS_BATCH_DIR = "/project/logs/batches"

MODEL_PATH = os.path.join(BASE_DIR, "ml", "log_classifier_v2.joblib")
VECTORIZER_PATH = os.path.join(BASE_DIR, "ml", "log_vectorizer_v2.joblib")

os.makedirs(LOCAL_BATCH_DIR, exist_ok=True)

# =========================
# Spark Session
# =========================
@st.cache_resource
def get_spark():
    return SparkSession.builder.appName("BatchLogAnalytics").getOrCreate()

# =========================
# Load ML
# =========================
@st.cache_resource
def load_ml():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer

# =========================
# Helpers
# =========================
def run_cmd(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)

def load_all_batches():
    spark = get_spark()
    return spark.read.json(f"hdfs://{HDFS_BATCH_DIR}/*.json")

def load_latest_batch(path):
    spark = get_spark()
    return spark.read.json(f"hdfs://{path}")

def clean_text(text: str) -> str:
    text = re.sub(r"[\|\=\{\}\[\]\:\,\"]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()

# =========================
# APP
# =========================
st.set_page_config(page_title="Batch Near Real-Time Log Analytics", layout="wide")
st.title("🚀 Batch-based Near Real-Time Log Analytics (Spark + ML)")

spark = get_spark()
model, vectorizer = load_ml()

# =========================
# SIDEBAR CONTROLS
# =========================
st.sidebar.header("Controls")

log_count = st.sidebar.number_input(
    "How many logs to generate in this batch?",
    min_value=50,
    max_value=100000,
    value=500,
    step=50
)

if "latest_batch_path" not in st.session_state:
    st.session_state.latest_batch_path = None

if st.sidebar.button("Generate New Batch"):
    with st.spinner("Generating new batch..."):

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_name = f"batch_{timestamp}.json"

        local_batch_path = os.path.join(LOCAL_BATCH_DIR, batch_name)
        hdfs_batch_path = f"{HDFS_BATCH_DIR}/{batch_name}"

        # 1. Generate logs (DEFAULT FILE)
        cmd_gen = f"python3 {GEN_SCRIPT} --total {log_count}"
        result = run_cmd(cmd_gen)

        if result.returncode != 0:
            st.error("❌ Log generation failed")
            st.text(result.stderr)
            st.stop()

        if not os.path.exists(DEFAULT_GEN_FILE):
            st.error("❌ enhanced_logs.json not created.")
            st.stop()

        # 2. Rename to batch file
        os.rename(DEFAULT_GEN_FILE, local_batch_path)
        st.success(f"✅ Batch created: {batch_name}")

        # 3. Ensure HDFS folder exists
        run_cmd(f"hdfs dfs -mkdir -p {HDFS_BATCH_DIR}")

        # 4. Upload to HDFS
        upload = run_cmd(f"hdfs dfs -put -f {local_batch_path} {hdfs_batch_path}")

        if upload.returncode != 0:
            st.error("❌ Upload to HDFS failed")
            st.text(upload.stderr)
            st.stop()

        st.success("✅ Uploaded batch to HDFS")

        # 5. Save latest batch path
        st.session_state.latest_batch_path = hdfs_batch_path
        st.stop()

# =========================
# SECTION 1 — HISTORICAL VIEW
# =========================
st.header("📊 Overall (Historical) Analytics — All Batches")

try:
    df_all = load_all_batches()
    total_logs = df_all.count()

    st.metric("Total Logs in HDFS (All Batches)", total_logs)

    st.subheader("Status Code Distribution")
    status_df = (
        df_all.groupBy("status_code")
        .count()
        .orderBy("count", ascending=False)
        .toPandas()
    )

    st.bar_chart(status_df.set_index("status_code"))
    st.dataframe(status_df)

    st.subheader("Top Endpoints (All Batches)")
    ep_df = (
        df_all.groupBy("endpoint")
        .count()
        .orderBy("count", ascending=False)
        .limit(10)
        .toPandas()
    )
    st.table(ep_df)

    st.subheader("Top IPs (All Batches)")
    ip_df = (
        df_all.groupBy("ip_address")
        .count()
        .orderBy("count", ascending=False)
        .limit(10)
        .toPandas()
    )
    st.table(ip_df)

except Exception:
    st.warning("No batches found in HDFS. Please generate the first batch.")

# =========================
# SECTION 2 — LATEST BATCH VIEW
# =========================
st.header("🆕 Latest Batch Analytics + ML Detection")

if st.session_state.latest_batch_path is None:
    st.info("No new batch generated in this session.")
else:
    st.code(f"Latest batch file: {st.session_state.latest_batch_path}")

    df_latest = load_latest_batch(st.session_state.latest_batch_path)
    latest_count = df_latest.count()

    st.metric("Logs in Latest Batch", latest_count)

    # ---------- Latest batch analytics ----------
    st.subheader("Latest Batch Status Codes")
    latest_status = (
        df_latest.groupBy("status_code")
        .count()
        .orderBy("count", ascending=False)
        .toPandas()
    )

    st.bar_chart(latest_status.set_index("status_code"))
    st.dataframe(latest_status)

    st.subheader("Latest Batch Top Endpoints")
    latest_ep = (
        df_latest.groupBy("endpoint")
        .count()
        .orderBy("count", ascending=False)
        .limit(10)
        .toPandas()
    )
    st.table(latest_ep)

    # ---------- ML on Latest Batch ----------
    st.subheader("🤖 ML Prediction on Latest Batch (Only)")

    pdf = df_latest.limit(1000).toPandas()

    raw_texts = []
    true_labels = []

    for _, row in pdf.iterrows():
        record = row.to_dict()
        label = int(record.get("label", 0))
        true_labels.append(label)
        record.pop("label", None)
        raw_texts.append(json.dumps(record))

    cleaned = [clean_text(t) for t in raw_texts]
    X = vectorizer.transform(cleaned)

    preds = model.predict(X)

    suspicious_pred = int(preds.sum())
    normal_pred = int(len(preds) - suspicious_pred)
    suspicious_true = int(sum(true_labels))

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted Suspicious", suspicious_pred)
    c2.metric("Predicted Normal", normal_pred)
    c3.metric("True Suspicious", suspicious_true)

    pred_df = pd.DataFrame({
        "Class": ["NORMAL", "SUSPICIOUS"],
        "Count": [normal_pred, suspicious_pred]
    }).set_index("Class")

    st.subheader("Prediction Distribution (Latest Only)")
    st.bar_chart(pred_df)

    pdf["true_label"] = ["SUSPICIOUS" if x == 1 else "NORMAL" for x in true_labels]
    pdf["predicted_label"] = ["SUSPICIOUS" if x == 1 else "NORMAL" for x in preds]

    st.subheader("Sample Results (Latest Batch)")
    st.dataframe(
        pdf[[
            "timestamp",
            "ip_address",
            "endpoint",
            "status_code",
            "true_label",
            "predicted_label"
        ]]
    )

# =========================
# FOOTER
# =========================
st.markdown("""
---
✅ Batch-based real-time simulation  
✅ Old data preserved  
✅ New batch added as new file  
✅ ML runs ONLY on newest logs  
✅ Perfect for viva + demo
""")
