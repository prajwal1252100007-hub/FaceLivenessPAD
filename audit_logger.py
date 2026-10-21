import os
from datetime import datetime

BASE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE, "logs")
LOG_FILE = os.path.join(LOG_DIR, "audit.log")


def log_event(role, event):
    os.makedirs(LOG_DIR, exist_ok=True)

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(
            f"{timestamp} | {role} | {event}\n"
        )


if __name__ == "__main__":
    log_event(
        "SYSTEM",
        "Audit logger test"
    )

    print("Audit logger working.")
    print(f"Log file: {LOG_FILE}")