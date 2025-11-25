# /home/hadoop/log_project_v2/data/enhanced_log_generator_v2.py

import random
import json
import os
from datetime import datetime, timedelta
import argparse

# ==========================
# CONFIG
# ==========================
DEFAULT_TOTAL_LOGS = 1000        # change if you want
SUSPICIOUS_PERCENT = 0.15          # 15% suspicious
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "enhanced_logs.json")

# === Enhanced Data Components ===
users = {
    "alice": {"role": "admin", "department": "IT"},
    "bob": {"role": "user", "department": "Sales"},
    "charlie": {"role": "manager", "department": "HR"},
    "david": {"role": "user", "department": "Engineering"},
    "eve": {"role": "analyst", "department": "Marketing"},
    "root": {"role": "superadmin", "department": "System"},
    "ubuntu": {"role": "service", "department": "Infrastructure"},
    "apache": {"role": "service", "department": "Web"},
    "mysql": {"role": "service", "department": "Database"},
    "nginx": {"role": "service", "department": "Proxy"}
}

ip_ranges = {
    "US": ["192.168.1.{}", "10.0.1.{}", "172.16.1.{}"],
    "EU": ["95.85.{}", "188.114.{}", "213.136.{}"],
    "ASIA": ["103.25.{}", "112.215.{}", "180.215.{}"]
}

suspicious_ips = [
    "45.83.23.10", "182.75.92.34", "103.44.12.8", "188.201.90.55",
    "223.178.24.11", "94.102.51.78", "185.220.101.132", "91.121.82.114"
]

api_endpoints = [
    {"path": "/api/v1/login", "method": "POST", "normal_response_time": (100, 500)},
    {"path": "/api/v1/users", "method": "GET", "normal_response_time": (50, 200)},
    {"path": "/api/v1/users/{id}", "method": "GET", "normal_response_time": (30, 150)},
    {"path": "/api/v1/products", "method": "GET", "normal_response_time": (80, 300)},
    {"path": "/api/v1/orders", "method": "POST", "normal_response_time": (200, 800)},
    {"path": "/api/v1/payments", "method": "POST", "normal_response_time": (300, 1200)},
    {"path": "/api/v1/admin/users", "method": "GET", "normal_response_time": (100, 400)},
    {"path": "/api/v1/config", "method": "PUT", "normal_response_time": (150, 600)}
]

error_types = {
    "400": "Bad Request",
    "401": "Unauthorized",
    "403": "Forbidden",
    "404": "Not Found",
    "500": "Internal Server Error",
    "502": "Bad Gateway",
    "503": "Service Unavailable"
}

devices = ["Desktop", "Mobile", "Tablet"]
browsers = ["Chrome", "Firefox", "Safari", "Edge", "Opera"]
os_list = ["Windows 10", "Windows 11", "macOS", "Linux", "iOS", "Android"]

referrers = [
    "https://www.google.com",
    "https://www.bing.com",
    "https://www.facebook.com",
    "https://www.twitter.com",
    "https://www.linkedin.com",
    "direct",
    "email-campaign",
    "internal"
]

suspicious_patterns = [
    {"type": "brute_force", "action": "multiple failed logins", "threshold": 5},
    {"type": "data_scraping", "action": "high-frequency API calls", "threshold": 1000},
    {"type": "unauthorized_access", "action": "accessed admin endpoints", "threshold": 1},
    {"type": "injection_attempt", "action": "SQL injection detected", "threshold": 1},
    {"type": "malicious_download", "action": "downloaded executable", "threshold": 1},
    {"type": "port_scanning", "action": "multiple port connections", "threshold": 10},
    {"type": "credential_stuffing", "action": "login with common passwords", "threshold": 3}
]


def generate_ip(region=None):
    if not region:
        region = random.choice(list(ip_ranges.keys()))
    base = random.choice(ip_ranges[region])
    if "{}" in base:
        return base.format(random.randint(1, 255))
    return base


def generate_user_agent():
    browser = random.choice(browsers)
    os_name = random.choice(os_list)
    device = random.choice(devices)
    version = f"{random.randint(1, 120)}.0.{random.randint(1000, 9999)}.100"
    return f"Mozilla/5.0 ({device}; {os_name}) AppleWebKit/537.36 (KHTML, like Gecko) {browser}/{version}"


def generate_response_time(endpoint, is_error=False):
    base_range = endpoint["normal_response_time"]
    if is_error:
        return random.randint(base_range[1] + 100, base_range[1] + 2000)
    return random.randint(base_range[0], base_range[1])


def generate_log_entry(is_suspicious=False):
    # Timestamp: more traffic in business hours
    hour = random.randint(0, 23)
    if 9 <= hour <= 17:
        timestamp = datetime.now() - timedelta(
            days=random.randint(0, 7),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )
    else:
        timestamp = datetime.now() - timedelta(
            days=random.randint(0, 30),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )

    user, user_info = random.choice(list(users.items()))
    endpoint = random.choice(api_endpoints)

    is_error = random.random() < 0.05
    error_code = random.choice(list(error_types.keys())) if is_error else "200"

    response_time = generate_response_time(endpoint, is_error)

    if is_suspicious and random.random() < 0.7:
        ip = random.choice(suspicious_ips)
    else:
        ip = generate_ip()

    device = random.choice(devices)
    browser = random.choice(browsers)
    user_agent = generate_user_agent()
    referrer = random.choice(referrers)

    log_data = {
        "timestamp": timestamp.strftime('%Y-%m-%d %H:%M:%S'),
        "user": user,
        "user_role": user_info["role"],
        "department": user_info["department"],
        "ip_address": ip,
        "method": endpoint["method"],
        "endpoint": endpoint["path"].format(id=random.randint(1, 1000)),
        "status_code": error_code,
        "response_time": response_time,
        "user_agent": user_agent,
        "device": device,
        "browser": browser,
        "referrer": referrer,
        "bytes_sent": random.randint(100, 50000),
        "bytes_received": random.randint(50, 10000)
    }

    label = 1 if is_suspicious else 0

    if is_suspicious:
        pattern = random.choice(suspicious_patterns)
        log_data["suspicious_pattern"] = pattern["type"]
        log_data["suspicious_action"] = pattern["action"]
        log_data["severity"] = random.choice(["low", "medium", "high", "critical"])

        if pattern["type"] == "brute_force":
            log_data["failed_attempts"] = random.randint(5, 20)
        elif pattern["type"] == "data_scraping":
            log_data["requests_per_minute"] = random.randint(1000, 5000)
        elif pattern["type"] == "injection_attempt":
            log_data["malicious_payload"] = "SELECT * FROM users WHERE 1=1 OR '1'='1'"

    # store label inside JSON
    log_data["label"] = label
    return log_data, label


def main(total_logs: int):
    normal_count = int(total_logs * (1 - SUSPICIOUS_PERCENT))
    suspicious_count = total_logs - normal_count

    print(f"Generating {total_logs:,} enhanced JSON logs in {OUTPUT_FILE} ...")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(OUTPUT_FILE, "w") as f:
        # Normal logs
        for i in range(normal_count):
            log_data, _ = generate_log_entry(False)
            f.write(json.dumps(log_data) + "\n")
            if (i + 1) % 50000 == 0:
                print(f"Normal logs generated: {i + 1}")

        # Suspicious logs
        for j in range(suspicious_count):
            log_data, _ = generate_log_entry(True)
            f.write(json.dumps(log_data) + "\n")
            if (j + 1) % 50000 == 0:
                print(f"Suspicious logs generated: {j + 1}")

    print("✅ Completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--total", type=int, default=DEFAULT_TOTAL_LOGS,
                        help="Total number of logs to generate")
    args = parser.parse_args()
    main(args.total)
