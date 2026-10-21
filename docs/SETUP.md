# Setup & Installation Guide

This guide walks through deploying the fleet monitor on a Linux host that can
reach your Janus WebRTC server over the network.

## 1. System dependencies

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip build-essential
```

Playwright's headless Chromium needs its own OS-level dependencies, installed
automatically by `playwright install --with-deps chromium` (see step 3).

## 2. Clone and create a virtualenv

```bash
git clone https://github.com/<your-username>/stream-quality-monitor.git
cd stream-quality-monitor
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

## 3. Install Python dependencies

```bash
pip install -r requirements.txt
playwright install --with-deps chromium
```

## 4. Configure your streams

Copy the example config and point it at your own Janus deployment:

```bash
cp config/streams.example.json config/streams.json
```

Edit `config/streams.json`:

```json
{
  "probe_url": "https://your-janus-server.example.com/path/janus_streaming_probe.html",
  "streams": [
    { "id": "100001", "zoom_in_view": 1 }
  ]
}
```

- `probe_url` — where `probe/janus_streaming_probe.html` (or the
  `_token` variant) is hosted and reachable by the monitor's headless browser.
- `streams` — one entry per stream/mountpoint id to watch. `zoom_in_view` is
  optional and only used if a stream has two expected camera positions
  (zoom-in / zoom-out) that should both be treated as normal.

## 5. Required environment variables

| Variable | Required | Description |
|---|---|---|
| `JANUS_WS` | yes | WebSocket URL of your Janus server (`wss://...`) |
| `STREAMS_CONFIG` | yes | Path to your `streams.json` |
| `METRICS_PORT` | no (default `9109`) | Port for the Prometheus `/metrics` endpoint |
| `BASIC_USER` / `BASIC_PASS` | no | HTTP Basic auth for loading the probe page itself |
| `JANUS_API_SECRET` | no | Janus plugin API secret, if your deployment requires one |
| `JANUS_TOKEN_AUTH_SECRET` | token variant only | HMAC secret used to mint short-lived Janus auth tokens |

See the top of each script for the full list of tunable thresholds
(black/blur/freeze detection, ROI size, concurrency, sampling rate, etc.) —
every one of them has a sensible default and can be overridden via
environment variable.

## 6. Run it

```bash
export JANUS_WS="wss://your-janus-server.example.com/"
export STREAMS_CONFIG="$(pwd)/config/streams.json"
python janus_browser_quality_monitor_fleet.py
```

Metrics will be available at `http://127.0.0.1:9109/metrics` in Prometheus
exposition format, labeled per stream id.

## 7. Running as a long-lived service (optional)

A minimal `systemd` unit:

```ini
[Unit]
Description=Stream Quality Monitor
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=monitor
WorkingDirectory=/opt/stream-quality-monitor
EnvironmentFile=/etc/stream-quality-monitor/env
ExecStart=/opt/stream-quality-monitor/.venv/bin/python /opt/stream-quality-monitor/janus_browser_quality_monitor_fleet.py
Restart=always
RestartSec=2
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

Put the environment variables above into `/etc/stream-quality-monitor/env`
(mode `600`, owned by root) rather than hardcoding them anywhere, then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now stream-quality-monitor.service
sudo journalctl -u stream-quality-monitor.service -f
```

## Troubleshooting

- **No frames / stalled** — confirm `JANUS_WS` is reachable from the host
  running the monitor, and that the stream id exists on the Janus streaming
  plugin.
- **False "blurry" or "black" alerts** — tune `BLUR_LAPLACIAN_THR`,
  `BLACK_MEAN_THR`, `BLACK_STD_THR` for your camera's lighting conditions.
- **Playwright fails to launch Chromium** — re-run
  `playwright install --with-deps chromium`.
