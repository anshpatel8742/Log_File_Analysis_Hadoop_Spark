🚀 Log Analysis System v2 — Hadoop + Spark + Streamlit
📌 Project Overview

This project is a Big Data Log Analysis System built using:

Hadoop (HDFS) for distributed storage

Apache Spark for fast data processing

Streamlit for interactive visualization

Python for log generation and analysis

It analyzes large-scale web server logs in JSON format and provides useful insights such as:

Error frequency

Traffic patterns

Top IP addresses

Most accessed APIs

Request method distribution

Device & browser analytics

Server performance (response time)

Referrer analysis

Scalability with Hadoop + Spark

This project demonstrates big-data analytics skills and distributed system concepts.

📂 Project Structure
log_project_v2/
│
├── data/
│   ├── enhanced_log_generator_v2.py   # Log generator script
│   └── enhanced_logs.json             # Generated logs (local)
│
├── dashboard/
│   └── app_v2.py                       # Main Streamlit dashboard
│
├── ml/
│   ├── log_classifier_v2.joblib       # Pre-trained ML model
│   └── log_vectorizer_v2.joblib       # TF-IDF vectorizer
│
└── README.md

⚙️ System Workflow
1. Log Generation (Local)

enhanced_log_generator_v2.py generates large amounts of structured JSON logs with fields such as:

timestamp

ip_address

method

endpoint

status_code

device

browser

referrer

response_time

label (for ML, already trained)

Example command:

cd ~/log_project_v2/data
python3 enhanced_log_generator_v2.py --total 1000


This creates:

data/enhanced_logs.json

2. Upload Logs to HDFS
hdfs dfs -mkdir -p /project/logs
hdfs dfs -put -f ~/log_project_v2/data/enhanced_logs.json /project/logs/


Confirm:

hdfs dfs -ls /project/logs


You should see:

/project/logs/enhanced_logs.json

3. Run Dashboard (Spark + HDFS)

Go to dashboard folder:

cd ~/log_project_v2/dashboard
streamlit run app_v2.py


This dashboard reads data from:

hdfs:// /project/logs/enhanced_logs.json


and performs Spark-based analytics.

📊 Features Implemented (Using Spark)

This dashboard demonstrates:

✅ Error frequency and type
✅ Traffic peaks by hour (line graph)
✅ Top IP addresses
✅ Most accessed APIs
✅ Request type distribution (GET/POST/etc)
✅ Device and browser analytics
✅ Server performance (avg / min / max response time)
✅ Referrer source analysis
✅ Big Data processing on Hadoop + Spark
✅ Scalable architecture

All analytics are performed using Apache Spark on HDFS data, not local data.

🔥 Technologies Used
Tool	Purpose
Hadoop (HDFS)	Distributed file storage
Apache Spark	Fast parallel data processing
Streamlit	Web dashboard
Python	Log generation + logic
Joblib	Model loading
Pandas	Small sample processing
Linux (Ubuntu)	Environment
🧠 Academic Highlights

This project shows understanding of:

Big Data architecture

Distributed storage (HDFS)

Parallel processing (Spark)

Data visualization

Log analysis

System scalability

Perfect for:

Final Year Project

Big Data Lab

Viva / Demonstration

✅ How to Demonstrate (Short Steps)

Show HDFS:

hdfs dfs -ls /project/logs


Run Dash:

streamlit run app_v2.py


Explain:

Logs stored on HDFS

Spark used for analytics

Dynamic dashboard

Scalable to millions of logs

🎯 Summary

This system proves that large-scale log data can be efficiently processed and analyzed using Hadoop + Spark, with a clean and interactive Streamlit dashboard.

Status: ✅ Working
Version: v2
Author: Log Analysis System – CSE Project
