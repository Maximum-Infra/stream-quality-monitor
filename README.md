# Stream Quality Monitor

A fleet monitor for Janus WebRTC video streams. It headlessly watches each
stream in a browser (Playwright + Chromium), samples frames, and computes
per-stream video health metrics — exposed as a Prometheus `/metrics`
endpoint — so stream outages, freezes, blur, or black frames can be caught
and alerted on automatically instead of discovered by a human noticing a
dead feed.

> **Status:** Actively used and tested in a real deployment; not a finished
> "1.0" product. Expect some rough edges and manual tuning per camera/site.

## What it does

- Connects to a Janus `streaming` plugin mountpoint per stream via a
  browser-based WebRTC probe page, in a rotating/concurrency-limited worker
  pool (so one process can watch many streams without one browser tab per
  stream staying open forever).
- Samples video frames and runs lightweight CV checks against each one:
  - **Black frame detection** (mean/stddev luma thresholds)
  - **Blur detection** (Laplacian variance)
  - **Freeze detection** (adaptive, per-stream baseline motion threshold)
  - **Stall detection** (no frames received within a timeout)
- Optional diagnostics layered on top of the quality checks:
  - **ORB + RANSAC position tracking** — flags if the camera's framing shifted
    (pan/tilt/zoom) relative to a learned reference frame.
  - **Two-view learning** — for cameras that normally alternate between two
    expected positions (e.g. zoomed-in / zoomed-out), it learns both
    fingerprints and only alerts if the current frame doesn't match either.
- Exposes everything as labeled Prometheus gauges/counters (`stream="<id>"`)
  for dashboards and alerting (Grafana, Alertmanager, etc.).

## Project layout

```
janus_browser_quality_monitor_fleet.py        # main monitor
janus_browser_quality_monitor_fleet_token.py  # variant with signed Janus auth tokens
probe/janus_streaming_probe.html              # browser probe page the monitor drives
probe/janus_streaming_probe_token.html        # probe page variant supporting token auth
config/streams.example.json                   # example stream list config
docs/SETUP.md                                 # full install/deploy walkthrough
```

Two variants of the monitor and probe page are included:

- **Plain** (`janus_browser_quality_monitor_fleet.py` +
  `janus_streaming_probe.html`) — for Janus deployments using API-secret or
  no auth.
- **Token** (`*_token.py` + `*_token.html`) — adds HMAC-signed, short-lived
  Janus auth tokens (`JANUS_TOKEN_AUTH_SECRET`) for deployments that require
  per-session token auth, plus URL-credential redaction in logs.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
playwright install --with-deps chromium

cp config/streams.example.json config/streams.json
# edit config/streams.json with your Janus probe URL and stream ids

export JANUS_WS="wss://your-janus-server.example.com/"
export STREAMS_CONFIG="$(pwd)/config/streams.json"
python janus_browser_quality_monitor_fleet.py
```

Metrics are then served at `http://127.0.0.1:9109/metrics`.

See [`docs/SETUP.md`](docs/SETUP.md) for the full setup guide, all
environment variables, tuning notes, and a sample `systemd` service.

## Key metrics exported

| Metric | Meaning |
|---|---|
| `video_ok` / `video_bad` | 1/0 — overall stream health |
| `video_bad_reason_code` | 0 ok, 1 black, 2 blur, 3 frozen, 4 stalled, 5 error |
| `video_black`, `video_blurry`, `video_frozen`, `video_stalled` | individual, debounced flags |
| `video_last_frame_age_seconds` | seconds since the last good frame |
| `video_position_changed` | 1 if ORB tracking detects the camera framing shifted |
| `video_view_active` | learned zoom state: 1 zoom-in / 0 zoom-out / -1 unknown |
| `video_view_changed_unexpected` | 1 if the current view matches neither learned reference |

Full list in the `Gauge`/`Counter` definitions near the top of each script.

## Requirements

- Python 3.9+
- A reachable Janus `streaming` plugin WebSocket endpoint
- See [`requirements.txt`](requirements.txt) for Python packages

## License

MIT — see [`LICENSE`](LICENSE).
