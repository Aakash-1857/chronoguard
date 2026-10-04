# ChronoGuard v1.0

ChronoGuard is a comprehensive pipeline for live network packet ingestion, drift detection, and a Streamlit-based control panel dashboard. It implements a 3-thread architecture for packet capture, ONNX inference, and background retraining with LightGBM.

## Features

- **Live Packet Capture**: Scapy-based sniffer reading from network interfaces.
- **Drift Detection**: ADWIN-based statistical drift monitoring.
- **Automated Retraining**: Background LightGBM retraining triggered by drift.
- **Dashboard**: A read-only Streamlit Control Panel for monitoring throughput, drift, and system logs.

## Prerequisites

- Python 3.10+
- macOS or Linux
- Raw socket capture privileges for live traffic sniffing.

## Setup & Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/chronoguard.git
cd chronoguard

# Create a virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Running the Application

ChronoGuard provides a unified orchestrator script (`run.py`) to launch the core runtime and the Streamlit dashboard together.

### Run Locally (requires sudo for packet capture)

```bash
# Start the full application (Dashboard + Core Runtime)
sudo python run.py
```

*The dashboard will be available at `http://localhost:8501`*

### Run without Dashboard

```bash
sudo python run.py --no-dashboard
```

### Run using a loopback interface (dummy traffic)
```bash
CHRONOGUARD_INTERFACE=lo CHRONOGUARD_LOG_LEVEL=INFO sudo -E python run.py
```

## Dashboard Overview

The read-only Streamlit dashboard provides:
1. **Status Bar** — live throughput (flows/sec), current drift state, active model version.
2. **Feature Tracking Chart** — live metric line, ADWIN bounds corridor, and drift-fire vertical rules.
3. **System Log Stream** — live-tailing, monospace-rendered log output with semantic coloring.
4. **Model Lineage Table** — retrain history with PASS/FAIL badges, accuracy metrics, and timestamps.

## Live Deployment

A live demo of the dashboard is available at: [LIVE_DEMO_URL_HERE]

*(Note: The live version might run the dashboard only without the packet sniffing capabilities, depending on the hosting platform's networking privileges.)*
