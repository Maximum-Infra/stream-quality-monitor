#!/usr/bin/env python3
"""
janus_browser_quality_monitor_fleet.py

Rotate-mode fleet monitor for Janus WebRTC streams using Playwright headless Chromium.

- One Prometheus /metrics endpoint (all streams)
- Labeled metrics: stream="<id>"
- Concurrency-limited workers
- Rotation: open probe page, sample frames for ROTATE_WINDOW_SEC, close, move on

Config JSON via STREAMS_CONFIG env var:
{
  "probe_url": "https://.../Sandbox/janus_streaming_probe.html",
  "streams": [
      "1002021",
      {"id":"1002022"},
      {"id":"1002023", "zoom_in_view": 0}
  ]
}

Probe URL params supported:
  Newer probe:
    ?server=<wss-url>&stream=<id>&apiSecret=<optional>
  Older probe:
    ?ws=<wss-url>&id=<id>

This script appends BOTH pairs for compatibility.

FEATURES:
- Quality checks:
    black / blur / frozen / stalled + bad_reason_code
- ORB+RANSAC position diagnostics:
    video_position_* metrics + boolean video_position_changed (confidence gated)
- Two expected "views" per stream (zoom-in / zoom-out):
    - Learns up to 2 view fingerprints per stream while video is healthy (video_ok==1)
    - Switch between view0 and view1 is considered normal
    - Alerts only if current view is farther than the normal view0<->view1 switch by a margin
    - Exposes metrics video_view_* including video_view_changed_unexpected

IMPORTANT CHANGE (per your request):
- video_view_active is now a ZOOM STATE:
    1 = zoom-in
    0 = zoom-out
   -1 = unknown
  This uses streams.json per-stream mapping "zoom_in_view": 0 or 1.
- The old behavior ("which view matched best") is now exported as:
    video_view_index

NOTE:
- This file is spaces-only indentation (no tabs).
"""

import os
import time
import json
import base64
import asyncio
from collections import deque
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from urllib.parse import urlencode

import numpy as np
import cv2
from prometheus_client import start_http_server, Gauge, Counter
from playwright.async_api import async_playwright


# ------------------------
# Env / defaults
# ------------------------
STREAMS_CONFIG = os.environ.get("STREAMS_CONFIG", "/etc/janus-monitor/streams.json")

# HTTP Basic for loading the probe URL (NOT for WSS auth inside browser)
BASIC_USER = os.environ.get("BASIC_USER", "")
BASIC_PASS = os.environ.get("BASIC_PASS", "")

JANUS_WS = os.environ.get("JANUS_WS", "").strip()
METRICS_PORT = int(os.environ.get("METRICS_PORT", "9109"))

# Rotate behavior
CONCURRENCY = int(os.environ.get("CONCURRENCY", "12"))
ROTATE_WINDOW_SEC = float(os.environ.get("ROTATE_WINDOW_SEC", "15.0"))
ROTATE_IDLE_SEC = float(os.environ.get("ROTATE_IDLE_SEC", "0.2"))

# Sampling
SAMPLE_FPS = float(os.environ.get("SAMPLE_FPS", "2.0"))

VIDEO_SELECTOR = os.environ.get("VIDEO_SELECTOR", "#v").strip()
VIDEO_SELECTORS_ENV = os.environ.get("VIDEO_SELECTORS", "").strip()
ROTATE = os.environ.get("ROTATE", "NONE").upper()

# Quality thresholds / ROI
USE_CENTER_ROI = os.environ.get("USE_CENTER_ROI", "1") == "1"
ROI_W = float(os.environ.get("ROI_W", "0.7"))
ROI_H = float(os.environ.get("ROI_H", "0.7"))

BLACK_MEAN_THR = float(os.environ.get("BLACK_MEAN_THR", "12.0"))
BLACK_STD_THR = float(os.environ.get("BLACK_STD_THR", "8.0"))
BLUR_LAPLACIAN_THR = float(os.environ.get("BLUR_LAPLACIAN_THR", "80.0"))

# Freeze detection (adaptive per-stream)
FREEZE_WINDOW_SEC = float(os.environ.get("FREEZE_WINDOW_SEC", "3.0"))
FREEZE_MIN_SAMPLES = int(os.environ.get("FREEZE_MIN_SAMPLES", "5"))
FREEZE_MULTIPLIER = float(os.environ.get("FREEZE_MULTIPLIER", "0.6"))
FREEZE_MIN_MEDIAN = float(os.environ.get("FREEZE_MIN_MEDIAN", "0.002"))
FREEZE_BASELINE_KEEP = int(os.environ.get("FREEZE_BASELINE_KEEP", "120"))

# Stall detection (within a stream visit)
NO_FRAME_STALL_SEC = float(os.environ.get("NO_FRAME_STALL_SEC", "3.0"))

# Persistence within visit (cheap debounce)
PERSIST_SEC = float(os.environ.get("PERSIST_SEC", "3.0"))

# Snapshots (content frame snapshots on bad)
SNAP_ON_BAD = os.environ.get("SNAP_ON_BAD", "1") == "1"
SNAP_BASE_DIR = Path(os.environ.get("SNAP_BASE_DIR", "/var/lib/janus-monitor/snaps"))
SNAP_COOLDOWN_SEC = float(os.environ.get("SNAP_COOLDOWN_SEC", "60.0"))

# Debug artifacts
DEBUG_DIR = Path(os.environ.get("DEBUG_DIR", "/tmp/janus-monitor-fleet"))
DEBUG_SCREENSHOT_ON_ERROR = os.environ.get("DEBUG_SCREENSHOT_ON_ERROR", "1") == "1"

# Sharding (optional)
SHARD_INDEX = int(os.environ.get("SHARD_INDEX", "0"))
SHARD_TOTAL = int(os.environ.get("SHARD_TOTAL", "1"))

# Backoff per stream after visit errors (prevents hammering)
VISIT_BACKOFF_MIN_SEC = float(os.environ.get("VISIT_BACKOFF_MIN_SEC", "2.0"))
VISIT_BACKOFF_MAX_SEC = float(os.environ.get("VISIT_BACKOFF_MAX_SEC", "30.0"))
VISIT_BACKOFF_MULT = float(os.environ.get("VISIT_BACKOFF_MULT", "1.7"))


# ------------------------
# ORB position-change (diagnostic)
# ------------------------
POS_ENABLE = os.environ.get("POS_ENABLE", "1") == "1"
POS_LEARN_MIN_GOOD_FRAMES = int(os.environ.get("POS_LEARN_MIN_GOOD_FRAMES", "6"))
POS_ORB_FEATURES = int(os.environ.get("POS_ORB_FEATURES", "800"))
POS_MIN_MATCHES = int(os.environ.get("POS_MIN_MATCHES", "25"))
POS_MAX_MATCHES_USE = int(os.environ.get("POS_MAX_MATCHES_USE", "80"))
POS_W = int(os.environ.get("POS_W", "480"))
POS_H = int(os.environ.get("POS_H", "270"))

POS_TRANS_THR_PX = float(os.environ.get("POS_TRANS_THR_PX", "25"))
POS_ROT_THR_DEG = float(os.environ.get("POS_ROT_THR_DEG", "6"))
POS_SCALE_THR = float(os.environ.get("POS_SCALE_THR", "0.06"))

POS_MIN_INLIERS = int(os.environ.get("POS_MIN_INLIERS", "18"))
POS_MIN_INLIER_RATIO = float(os.environ.get("POS_MIN_INLIER_RATIO", "0.35"))
POS_DETECT_MIN_CONSEC = int(os.environ.get("POS_DETECT_MIN_CONSEC", "3"))
POS_HOLD_SEC = float(os.environ.get("POS_HOLD_SEC", "10.0"))

POS_REF_DIR = Path(os.environ.get("POS_REF_DIR", "/var/lib/janus-monitor/posref"))


# ------------------------
# Two-view learning (primary for zoom-in/zoom-out)
# ------------------------
VIEW_ENABLE = os.environ.get("VIEW_ENABLE", "1") == "1"
VIEW_REF_DIR = Path(os.environ.get("VIEW_REF_DIR", "/var/lib/janus-monitor/viewref"))

VIEW_W = int(os.environ.get("VIEW_W", "320"))
VIEW_H = int(os.environ.get("VIEW_H", "180"))
VIEW_CANNY1 = int(os.environ.get("VIEW_CANNY1", "60"))
VIEW_CANNY2 = int(os.environ.get("VIEW_CANNY2", "160"))
VIEW_HASH_W = int(os.environ.get("VIEW_HASH_W", "8"))
VIEW_HASH_H = int(os.environ.get("VIEW_HASH_H", "8"))

VIEW_LEARN_MIN_GOOD_FRAMES = int(os.environ.get("VIEW_LEARN_MIN_GOOD_FRAMES", "8"))
VIEW_LEARN_SECOND_MIN_DIST = int(os.environ.get("VIEW_LEARN_SECOND_MIN_DIST", "18"))

VIEW2_CANDIDATE_WINDOW = int(os.environ.get("VIEW2_CANDIDATE_WINDOW", "24"))
VIEW2_CANDIDATE_MIN = int(os.environ.get("VIEW2_CANDIDATE_MIN", "10"))

VIEW_MATCH_MAX_DIST = int(os.environ.get("VIEW_MATCH_MAX_DIST", "12"))
VIEW_ALERT_MULT = float(os.environ.get("VIEW_ALERT_MULT", "1.35"))
VIEW_ALERT_MARGIN = int(os.environ.get("VIEW_ALERT_MARGIN", "6"))

VIEW_UNEXPECTED_CONSEC = int(os.environ.get("VIEW_UNEXPECTED_CONSEC", "3"))
VIEW_HOLD_SEC = float(os.environ.get("VIEW_HOLD_SEC", "15.0"))


# ------------------------
# Prometheus metrics (labeled)
# ------------------------
LABELS = ["stream"]

g_monitor_up = Gauge("video_monitor_up", "1 if worker is actively monitoring this stream (best-effort)", LABELS)

g_ok = Gauge("video_ok", "1 if OK else 0", LABELS)
g_bad = Gauge("video_bad", "1 if bad else 0", LABELS)
g_reason = Gauge("video_bad_reason_code", "0 ok, 1 black, 2 blur, 3 frozen, 4 stalled, 5 error", LABELS)

g_black = Gauge("video_black", "1 if black detected (persisted)", LABELS)
g_blur = Gauge("video_blurry", "1 if blur detected (persisted)", LABELS)
g_frozen = Gauge("video_frozen", "1 if freeze detected (persisted)", LABELS)
g_stalled = Gauge("video_stalled", "1 if no frames extracted (persisted)", LABELS)

g_luma_mean = Gauge("video_luma_mean", "mean grayscale luma (ROI)", LABELS)
g_luma_std = Gauge("video_luma_std", "stddev grayscale luma (ROI)", LABELS)
g_lap_var = Gauge("video_laplacian_var", "laplacian variance (blur score)", LABELS)

g_frames_seen = Gauge("video_frames_seen", "frames sampled (lifetime, per process)", LABELS)
g_last_frame_age = Gauge("video_last_frame_age_seconds", "seconds since last successful frame", LABELS)
g_last_visit = Gauge("video_last_visit_unix", "unix timestamp of last rotation visit", LABELS)

c_visits_total = Counter("video_visits_total", "Total rotation visits per stream", LABELS)
c_visit_errors_total = Counter("video_visit_errors_total", "Visit errors per stream", LABELS)

# ORB position metrics
g_pos_changed = Gauge("video_position_changed", "1 if camera/view changed vs learned ORB reference", LABELS)
g_pos_dx = Gauge("video_position_dx_pixels", "Estimated dx vs ORB reference (pixels)", LABELS)
g_pos_dy = Gauge("video_position_dy_pixels", "Estimated dy vs ORB reference (pixels)", LABELS)
g_pos_rot = Gauge("video_position_rot_deg", "Estimated rotation vs ORB reference (degrees)", LABELS)
g_pos_scale = Gauge("video_position_scale", "Estimated scale vs ORB reference", LABELS)
g_pos_matches = Gauge("video_position_match_count", "ORB matches used (post-trim)", LABELS)
g_pos_inliers = Gauge("video_position_inlier_count", "RANSAC inliers for ORB affine", LABELS)
g_pos_inlier_ratio = Gauge("video_position_inlier_ratio", "inliers/matches ratio for ORB", LABELS)
g_pos_ref_age = Gauge("video_position_reference_age_seconds", "seconds since ORB reference was last set/loaded", LABELS)

# View metrics
g_view_known = Gauge("video_view_known_count", "How many expected views are learned (0..2)", LABELS)

# video_view_active is now zoom-state:
#   1 = Zoom-in
#   0 = Zoom-out
#  -1 = Unknown (not learned yet, or not configured)
g_view_active = Gauge(
    "video_view_active",
    "Zoom state derived from learned views: 1 zoom-in / 0 zoom-out / -1 unknown",
    LABELS,
)

# Old meaning preserved:
#   0/1 = which learned view matched best
#  -1  = unknown
g_view_index = Gauge(
    "video_view_index",
    "Which learned view matches best: 0/1, or -1 unknown",
    LABELS,
)

g_view_dist = Gauge("video_view_distance", "Hamming distance to closest learned view", LABELS)
g_view_switch_dist = Gauge("video_view_switch_distance", "Hamming distance between learned view0 and view1", LABELS)
g_view_unexpected = Gauge("video_view_changed_unexpected", "1 if view doesn't match either learned view beyond allowed threshold", LABELS)
g_view_dist0 = Gauge("video_view_distance_to_0", "Hamming distance to learned view0", LABELS)
g_view_dist1 = Gauge("video_view_distance_to_1", "Hamming distance to learned view1", LABELS)

# ------------------------
# Helpers
# ------------------------
def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{ts} {msg}", flush=True)


def now_ts_str() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def ensure_dir(p: Path) -> None:
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def rotate_frame(img: np.ndarray) -> np.ndarray:
    if ROTATE == "90CW":
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if ROTATE == "90CCW":
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if ROTATE == "180":
        return cv2.rotate(img, cv2.ROTATE_180)
    return img


def center_roi(frame: np.ndarray) -> np.ndarray:
    if not USE_CENTER_ROI:
        return frame
    h, w = frame.shape[:2]
    cw = max(1, int(w * ROI_W))
    ch = max(1, int(h * ROI_H))
    x0 = (w - cw) // 2
    y0 = (h - ch) // 2
    return frame[y0:y0 + ch, x0:x0 + cw]


def laplacian_var(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def preprocess_for_freeze(frame: np.ndarray) -> np.ndarray:
    small = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray


def diff_norm(a: np.ndarray, b: np.ndarray) -> float:
    return float(cv2.absdiff(a, b).mean()) / 255.0


def build_probe_url(probe_url: str, janus_ws: str, stream_id: str) -> str:
    params = {
        "server": janus_ws,
        "stream": stream_id,
        "ws": janus_ws,
        "id": stream_id,
    }
    api_secret = os.environ.get("JANUS_API_SECRET", "").strip()
    if api_secret:
        params["apiSecret"] = api_secret

    sep = "&" if "?" in probe_url else "?"
    return f"{probe_url}{sep}{urlencode(params)}"


def shard_streams(streams: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if SHARD_TOTAL <= 1:
        return streams
    out: List[Dict[str, Any]] = []
    for i, s in enumerate(streams):
        if (i % SHARD_TOTAL) == SHARD_INDEX:
            out.append(s)
    return out


def load_config(path: str) -> Tuple[str, List[Dict[str, Any]]]:
    try:
        raw = Path(path).read_text(encoding="utf-8")
    except Exception as e:
        raise SystemExit(f"Failed reading STREAMS_CONFIG={path}: {type(e).__name__}: {e}")

    try:
        cfg = json.loads(raw)
    except json.JSONDecodeError as e:
        # Print a small context window around the error location
        lines = raw.splitlines()
        bad_line = e.lineno
        start = max(1, bad_line - 3)
        end = min(len(lines), bad_line + 3)
        snippet = "\n".join(f"{i:4d}: {lines[i-1]}" for i in range(start, end + 1))
        raise SystemExit(
            f"Invalid JSON in STREAMS_CONFIG={path}: {e}\n"
            f"Context:\n{snippet}\n"
            f"Tip: remove comments and trailing commas."
        )

    probe_url = cfg.get("probe_url")
    if not probe_url:
        raise SystemExit("streams.json must include 'probe_url'")

    streams_raw = cfg.get("streams", [])
    streams: List[Dict[str, Any]] = []

    if isinstance(streams_raw, list):
        for x in streams_raw:
            if isinstance(x, str):
                streams.append({"id": x})
            elif isinstance(x, dict) and "id" in x:
                item: Dict[str, Any] = {"id": str(x["id"])}
                if "zoom_in_view" in x:
                    try:
                        item["zoom_in_view"] = int(x["zoom_in_view"])
                    except Exception:
                        pass
                streams.append(item)

    if not streams:
        raise SystemExit("streams.json must include non-empty 'streams' list")

    return probe_url, shard_streams(streams)


def parse_selectors() -> List[str]:
    selectors: List[str] = []
    if VIDEO_SELECTOR:
        selectors.append(VIDEO_SELECTOR)

    if VIDEO_SELECTORS_ENV:
        for s in VIDEO_SELECTORS_ENV.split(","):
            s = s.strip()
            if s:
                selectors.append(s)

    for s in ["#v", "#remoteVideo", "#remotevideo", "video"]:
        if s not in selectors:
            selectors.append(s)

    return selectors


async def debug_dump_page(page, stream_id: str, tag: str) -> None:
    try:
        url = page.url
    except Exception:
        url = "<unknown>"

    try:
        title = await page.title()
    except Exception:
        title = "<unknown>"

    snippet = ""
    try:
        html = await page.content()
        snippet = (html or "")[:800].replace("\n", " ").replace("\r", " ")
    except Exception:
        snippet = "<content unavailable>"

    log(f"[debug] stream={stream_id} tag={tag} url={url} title={title} html_snippet={snippet}")

    if DEBUG_SCREENSHOT_ON_ERROR:
        try:
            ensure_dir(DEBUG_DIR)
            out = DEBUG_DIR / f"{now_ts_str()}_{stream_id}_{tag}.png"
            await page.screenshot(path=str(out), full_page=True)
            log(f"[debug] stream={stream_id} saved_screenshot={out}")
        except Exception as e:
            log(f"[debug] stream={stream_id} screenshot_failed={type(e).__name__}: {e}")


# ------------------------
# ORB helpers (diagnostic)
# ------------------------
def preprocess_for_position(frame_bgr: np.ndarray) -> np.ndarray:
    small = cv2.resize(frame_bgr, (POS_W, POS_H), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray


def pos_ref_path(stream_id: str) -> Path:
    return POS_REF_DIR / f"stream_{stream_id}_orbref.npz"


def pos_save_ref(stream_id: str, ref_gray: np.ndarray, kp, des: np.ndarray) -> None:
    try:
        ensure_dir(POS_REF_DIR)
        pts = np.array([k.pt for k in kp], dtype=np.float32) if kp else np.zeros((0, 2), dtype=np.float32)
        np.savez_compressed(
            str(pos_ref_path(stream_id)),
            ref_gray=ref_gray,
            kp_pts=pts,
            des=des if des is not None else np.zeros((0, 32), dtype=np.uint8),
            ts_unix=np.array([int(time.time())], dtype=np.int64),
        )
    except Exception:
        pass


def pos_load_ref(stream_id: str):
    p = pos_ref_path(stream_id)
    if not p.exists():
        return None
    try:
        data = np.load(str(p), allow_pickle=False)
        ref_gray = data["ref_gray"]
        kp_pts = data["kp_pts"]
        des = data["des"]
        ts_unix = int(data["ts_unix"][0]) if "ts_unix" in data else 0
        kp = [cv2.KeyPoint(float(x), float(y), 31) for x, y in kp_pts]  # size arbitrary
        return ref_gray, kp, des, ts_unix
    except Exception:
        return None


def estimate_position_change(
    orb,
    ref_gray: np.ndarray,
    ref_kp,
    ref_des: np.ndarray,
    cur_gray: np.ndarray,
) -> Tuple[bool, float, float, float, float, int, int, float]:
    kp2, des2 = orb.detectAndCompute(cur_gray, None)
    if des2 is None or ref_des is None or kp2 is None or ref_kp is None:
        return (False, 0.0, 0.0, 0.0, 1.0, 0, 0, 0.0)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    try:
        matches = bf.match(ref_des, des2)
    except Exception:
        return (False, 0.0, 0.0, 0.0, 1.0, 0, 0, 0.0)

    if not matches:
        return (False, 0.0, 0.0, 0.0, 1.0, 0, 0, 0.0)

    matches = sorted(matches, key=lambda m: m.distance)
    matches = matches[: max(POS_MIN_MATCHES, min(POS_MAX_MATCHES_USE, len(matches)))]

    if len(matches) < POS_MIN_MATCHES:
        return (False, 0.0, 0.0, 0.0, 1.0, len(matches), 0, 0.0)

    pts1 = np.float32([ref_kp[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

    M, inliers = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if M is None or inliers is None:
        return (False, 0.0, 0.0, 0.0, 1.0, len(matches), 0, 0.0)

    inlier_count = int(inliers.sum())
    match_count = int(len(matches))
    inlier_ratio = float(inlier_count / max(1, match_count))

    if inlier_count < POS_MIN_INLIERS or inlier_ratio < POS_MIN_INLIER_RATIO:
        return (False, 0.0, 0.0, 0.0, 1.0, match_count, inlier_count, inlier_ratio)

    dx = float(M[0, 2])
    dy = float(M[1, 2])

    a = float(M[0, 0])
    b = float(M[0, 1])
    scale = float(np.sqrt(a * a + b * b))
    rot_deg = float(np.degrees(np.arctan2(b, a)))

    changed = (
        (abs(dx) > POS_TRANS_THR_PX)
        or (abs(dy) > POS_TRANS_THR_PX)
        or (abs(rot_deg) > POS_ROT_THR_DEG)
        or (abs(scale - 1.0) > POS_SCALE_THR)
    )

    return (changed, dx, dy, rot_deg, scale, match_count, inlier_count, inlier_ratio)


# ------------------------
# View fingerprint helpers
# ------------------------
def view_preprocess_edges(frame_bgr: np.ndarray) -> np.ndarray:
    small = cv2.resize(frame_bgr, (VIEW_W, VIEW_H), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, VIEW_CANNY1, VIEW_CANNY2)
    return edges


def dhash_bits(img_gray_or_edges: np.ndarray, hash_w: int, hash_h: int) -> np.ndarray:
    resized = cv2.resize(img_gray_or_edges, (hash_w + 1, hash_h), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    return diff.astype(np.uint8).reshape(-1)


def bits_to_hex(bits: np.ndarray) -> str:
    b = bits.astype(np.uint8)
    if (len(b) % 8) != 0:
        pad = 8 - (len(b) % 8)
        b = np.concatenate([b, np.zeros((pad,), dtype=np.uint8)])
    bytes_arr = np.packbits(b)
    return bytes_arr.tobytes().hex()


def hex_to_bits(hex_str: str, expected_bits: int) -> Optional[np.ndarray]:
    try:
        raw = bytes.fromhex(hex_str)
        arr = np.frombuffer(raw, dtype=np.uint8)
        bits = np.unpackbits(arr).astype(np.uint8)
        if len(bits) < expected_bits:
            return None
        return bits[:expected_bits]
    except Exception:
        return None


def hamming_dist(a_bits: np.ndarray, b_bits: np.ndarray) -> int:
    if a_bits is None or b_bits is None:
        return 999999
    if len(a_bits) != len(b_bits):
        n = min(len(a_bits), len(b_bits))
        return int(np.count_nonzero(a_bits[:n] ^ b_bits[:n])) + abs(len(a_bits) - len(b_bits))
    return int(np.count_nonzero(a_bits ^ b_bits))


def view_ref_path(stream_id: str) -> Path:
    return VIEW_REF_DIR / f"stream_{stream_id}_views.json"


def view_load_refs(stream_id: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    p = view_ref_path(stream_id)
    if not p.exists():
        return (None, None)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        exp_bits = VIEW_HASH_W * VIEW_HASH_H
        v0 = hex_to_bits(data.get("view0_hex", ""), exp_bits)
        v1 = hex_to_bits(data.get("view1_hex", ""), exp_bits)
        return (v0, v1)
    except Exception:
        return (None, None)


def view_save_refs(stream_id: str, v0: Optional[np.ndarray], v1: Optional[np.ndarray]) -> None:
    try:
        ensure_dir(VIEW_REF_DIR)
        out: Dict[str, Any] = {}
        if v0 is not None:
            out["view0_hex"] = bits_to_hex(v0)
        if v1 is not None:
            out["view1_hex"] = bits_to_hex(v1)
        out["hash_w"] = VIEW_HASH_W
        out["hash_h"] = VIEW_HASH_H
        out["ts_unix"] = int(time.time())
        view_ref_path(stream_id).write_text(json.dumps(out, indent=2), encoding="utf-8")
    except Exception:
        pass


def view_known_count(v0: Optional[np.ndarray], v1: Optional[np.ndarray]) -> int:
    return int(v0 is not None) + int(v1 is not None)


def view_switch_distance(v0: Optional[np.ndarray], v1: Optional[np.ndarray]) -> int:
    if v0 is None or v1 is None:
        return 0
    return hamming_dist(v0, v1)


def view_match(v0: Optional[np.ndarray], v1: Optional[np.ndarray], cur_bits: np.ndarray) -> Tuple[int, int]:
    best_idx = -1
    best_dist = 999999
    if v0 is not None:
        d0 = hamming_dist(v0, cur_bits)
        best_idx = 0
        best_dist = d0
    if v1 is not None:
        d1 = hamming_dist(v1, cur_bits)
        if d1 < best_dist:
            best_idx = 1
            best_dist = d1
    return (best_idx, int(best_dist))


def choose_medoid(bits_list: List[np.ndarray]) -> np.ndarray:
    if not bits_list:
        raise ValueError("empty bits_list")
    n = len(bits_list)
    if n == 1:
        return bits_list[0]

    best_i = 0
    best_sum = None
    for i in range(n):
        s = 0
        bi = bits_list[i]
        for j in range(n):
            if i == j:
                continue
            s += hamming_dist(bi, bits_list[j])
        if best_sum is None or s < best_sum:
            best_sum = s
            best_i = i
    return bits_list[best_i]


# ------------------------
# Per-stream state
# ------------------------
class StreamState:
    def __init__(self):
        self.frames_seen: int = 0
        self.last_frame_ts: float = 0.0
        self.last_snap_ts: float = 0.0
        self.baseline_diffs = deque(maxlen=FREEZE_BASELINE_KEEP)

        self.backoff_sec: float = VISIT_BACKOFF_MIN_SEC
        self.backoff_until: float = 0.0

        # ORB ref per stream
        self.pos_good_streak: int = 0
        self.pos_ref_gray: Optional[np.ndarray] = None
        self.pos_ref_kp = None
        self.pos_ref_des: Optional[np.ndarray] = None
        self.pos_changed_consec: int = 0
        self.pos_changed_until: float = 0.0
        self.pos_ref_ts: int = 0
        self.pos_loaded: bool = False

        # View refs per stream
        self.view_good_streak: int = 0
        self.view0_bits: Optional[np.ndarray] = None
        self.view1_bits: Optional[np.ndarray] = None
        self.view_loaded: bool = False
        self.view2_candidates: deque = deque(maxlen=VIEW2_CANDIDATE_WINDOW)

        self.view_unexpected_consec: int = 0
        self.view_unexpected_until: float = 0.0


# ------------------------
# Visit logic
# ------------------------
async def select_video_selector(page, selectors: List[str], timeout_ms: int) -> str:
    last_err: Optional[Exception] = None
    per_try = max(1000, int(timeout_ms / max(1, len(selectors))))

    for sel in selectors:
        try:
            await page.wait_for_selector(sel, timeout=per_try, state="attached")
            return sel
        except Exception as e:
            last_err = e
            continue

    if last_err:
        raise last_err
    raise TimeoutError("No selector matched")


def build_capture_script(selector: str) -> str:
    return f"""
    (function() {{
        const sel = {selector!r};
        const video = document.querySelector(sel);
        if(!video) return {{ok:false, msg: "no video element", selector: sel}};
        if(video.readyState < 2 || (video.videoWidth === 0 || video.videoHeight === 0)) {{
            return {{
                ok:false,
                msg: "video not ready",
                selector: sel,
                width: video.videoWidth,
                height: video.videoHeight,
                readyState: video.readyState
            }};
        }}
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const data = canvas.toDataURL('image/jpeg', 0.7);
        return {{ok:true, data: data, width: video.videoWidth, height: video.videoHeight}};
    }})();
    """


def maybe_load_pos_ref(stream_label: str, st: StreamState, orb) -> None:
    if not POS_ENABLE:
        return
    if st.pos_loaded:
        return
    st.pos_loaded = True
    loaded = pos_load_ref(stream_label)
    if loaded is None:
        return
    ref_gray, kp, des, ts_unix = loaded
    st.pos_ref_gray = ref_gray
    st.pos_ref_kp = kp
    st.pos_ref_des = des
    st.pos_ref_ts = ts_unix
    log(f"[pos] stream={stream_label} loaded ORB reference from disk (kp={len(kp) if kp else 0})")


def maybe_load_view_refs(stream_label: str, st: StreamState) -> None:
    if not VIEW_ENABLE:
        return
    if st.view_loaded:
        return
    st.view_loaded = True
    v0, v1 = view_load_refs(stream_label)
    st.view0_bits = v0
    st.view1_bits = v1
    if v0 is not None or v1 is not None:
        log(
            f"[view] stream={stream_label} loaded view refs from disk "
            f"(v0={'yes' if v0 is not None else 'no'} v1={'yes' if v1 is not None else 'no'})"
        )


async def monitor_one_visit(
    page,
    probe_url: str,
    stream_id: str,
    st: StreamState,
    sample_interval: float,
    selectors: List[str],
    orb,
    zoom_in_view: Optional[int],
) -> None:
    stream_label = str(stream_id)

    # Load refs (once per process lifetime)
    maybe_load_pos_ref(stream_label, st, orb)
    maybe_load_view_refs(stream_label, st)

    now = time.time()
    if st.backoff_until and now < st.backoff_until:
        g_last_visit.labels(stream=stream_label).set(float(now))
        g_monitor_up.labels(stream=stream_label).set(0.0)
        await asyncio.sleep(min(ROTATE_IDLE_SEC, max(0.1, st.backoff_until - now)))
        return

    visit_start = time.time()
    g_last_visit.labels(stream=stream_label).set(float(visit_start))
    c_visits_total.labels(stream=stream_label).inc()
    g_monitor_up.labels(stream=stream_label).set(1.0)

    if not st.last_frame_ts:
        st.last_frame_ts = visit_start

    # Default metrics at visit start
    g_ok.labels(stream=stream_label).set(0.0)
    g_bad.labels(stream=stream_label).set(1.0)
    g_reason.labels(stream=stream_label).set(5.0)
    g_stalled.labels(stream=stream_label).set(1.0)

    # Init position metrics
    g_pos_dx.labels(stream=stream_label).set(0.0)
    g_pos_dy.labels(stream=stream_label).set(0.0)
    g_pos_rot.labels(stream=stream_label).set(0.0)
    g_pos_scale.labels(stream=stream_label).set(1.0)
    g_pos_matches.labels(stream=stream_label).set(0.0)
    g_pos_inliers.labels(stream=stream_label).set(0.0)
    g_pos_inlier_ratio.labels(stream=stream_label).set(0.0)
    if st.pos_ref_ts:
        g_pos_ref_age.labels(stream=stream_label).set(float(max(0, int(time.time()) - st.pos_ref_ts)))
    else:
        g_pos_ref_age.labels(stream=stream_label).set(0.0)
    if time.time() >= st.pos_changed_until:
        g_pos_changed.labels(stream=stream_label).set(0.0)

    # Init view metrics
    if VIEW_ENABLE:
        kc0 = view_known_count(st.view0_bits, st.view1_bits)
        g_view_known.labels(stream=stream_label).set(float(kc0))

        g_view_active.labels(stream=stream_label).set(-1.0)
        g_view_index.labels(stream=stream_label).set(-1.0)

        g_view_dist.labels(stream=stream_label).set(0.0)
        g_view_switch_dist.labels(stream=stream_label).set(float(view_switch_distance(st.view0_bits, st.view1_bits)))
        if time.time() >= st.view_unexpected_until:
            g_view_unexpected.labels(stream=stream_label).set(0.0)

    if not JANUS_WS:
        raise RuntimeError("JANUS_WS is empty; probe requires ws query param.")

    url = build_probe_url(probe_url, JANUS_WS, stream_id)
    await page.goto(url, wait_until="domcontentloaded", timeout=20000)

    chosen = await select_video_selector(page, selectors, timeout_ms=15000)
    capture_script = build_capture_script(chosen)

    persist_win = deque()  # (ts, black, blur, frozen, stalled)
    freeze_buf = deque()   # (ts, diff)
    last_gray = None

    while (time.time() - visit_start) < ROTATE_WINDOW_SEC:
        now = time.time()

        # Capture a frame
        res = None
        try:
            res = await page.evaluate(capture_script)
        except Exception:
            res = None

        if not res or not isinstance(res, dict) or not res.get("ok"):
            age = now - st.last_frame_ts
            g_last_frame_age.labels(stream=stream_label).set(float(age))

            stalled = age > NO_FRAME_STALL_SEC
            persist_win.append((now, False, False, False, stalled))
            while persist_win and (now - persist_win[0][0]) > max(PERSIST_SEC, NO_FRAME_STALL_SEC):
                persist_win.popleft()

            if stalled:
                g_ok.labels(stream=stream_label).set(0.0)
                g_bad.labels(stream=stream_label).set(1.0)
                g_reason.labels(stream=stream_label).set(4.0)
                g_stalled.labels(stream=stream_label).set(1.0)

            await asyncio.sleep(sample_interval)
            continue

        img = None
        try:
            _, b64 = res["data"].split(",", 1)
            jpg = base64.b64decode(b64)
            nparr = np.frombuffer(jpg, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception:
            img = None

        if img is None:
            age = now - st.last_frame_ts
            g_last_frame_age.labels(stream=stream_label).set(float(age))
            await asyncio.sleep(sample_interval)
            continue

        # Successful frame
        st.last_frame_ts = now
        g_last_frame_age.labels(stream=stream_label).set(0.0)

        img = rotate_frame(img)
        img = center_roi(img)

        gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean = float(gray_full.mean())
        std = float(gray_full.std())
        blur = float(laplacian_var(gray_full))

        raw_black = (mean < BLACK_MEAN_THR and std < BLACK_STD_THR)
        raw_blur = (blur < BLUR_LAPLACIAN_THR)

        # Freeze detection (adaptive baseline)
        gray_small = preprocess_for_freeze(img)
        raw_frozen = False
        if last_gray is not None:
            d = diff_norm(gray_small, last_gray)
            st.baseline_diffs.append(d)

            try:
                med = max(float(np.median(list(st.baseline_diffs))), FREEZE_MIN_MEDIAN)
            except Exception:
                med = FREEZE_MIN_MEDIAN

            freeze_thr = med * FREEZE_MULTIPLIER

            freeze_buf.append((now, d))
            while freeze_buf and (now - freeze_buf[0][0]) > FREEZE_WINDOW_SEC:
                freeze_buf.popleft()

            if len(freeze_buf) >= FREEZE_MIN_SAMPLES:
                if all(x < freeze_thr for _, x in freeze_buf):
                    raw_frozen = True

        last_gray = gray_small

        # Persist window (debounce)
        persist_win.append((now, raw_black, raw_blur, raw_frozen, False))
        while persist_win and (now - persist_win[0][0]) > PERSIST_SEC:
            persist_win.popleft()

        min_samples = max(2, int(PERSIST_SEC * SAMPLE_FPS * 0.8))

        def all_true(idx: int) -> bool:
            if len(persist_win) < min_samples:
                return False
            return all(x[idx] for x in persist_win)

        p_black = all_true(1)
        p_blur = all_true(2)
        p_frozen = all_true(3)
        p_stalled = all_true(4)

        # Reason priority
        reason = 0
        if p_stalled:
            reason = 4
        elif p_black:
            reason = 1
        elif p_frozen:
            reason = 3
        elif p_blur:
            reason = 2

        bad = 1.0 if reason != 0 else 0.0

        # Metrics
        st.frames_seen += 1
        g_frames_seen.labels(stream=stream_label).set(float(st.frames_seen))

        g_luma_mean.labels(stream=stream_label).set(mean)
        g_luma_std.labels(stream=stream_label).set(std)
        g_lap_var.labels(stream=stream_label).set(blur)

        g_black.labels(stream=stream_label).set(1.0 if p_black else 0.0)
        g_blur.labels(stream=stream_label).set(1.0 if p_blur else 0.0)
        g_frozen.labels(stream=stream_label).set(1.0 if p_frozen else 0.0)
        g_stalled.labels(stream=stream_label).set(1.0 if p_stalled else 0.0)

        g_reason.labels(stream=stream_label).set(float(reason))
        g_bad.labels(stream=stream_label).set(bad)
        g_ok.labels(stream=stream_label).set(1.0 if bad == 0.0 else 0.0)

        # Snapshot on bad (cooldown)
        if SNAP_ON_BAD and reason != 0 and (now - st.last_snap_ts) >= SNAP_COOLDOWN_SEC:
            st.last_snap_ts = now
            out_dir = SNAP_BASE_DIR / stream_label
            ensure_dir(out_dir)
            fname = f"{now_ts_str()}_stream{stream_label}_reason{reason}.jpg"
            try:
                cv2.imwrite(str(out_dir / fname), img)
            except Exception:
                pass

        # ------------------------
        # ORB position diagnostics (only when video OK)
        # ------------------------
        if POS_ENABLE:
            if reason == 0:
                st.pos_good_streak += 1
            else:
                st.pos_good_streak = 0
                st.pos_changed_consec = 0

            # Learn reference if none
            if st.pos_ref_gray is None and st.pos_good_streak >= POS_LEARN_MIN_GOOD_FRAMES:
                ref_gray = preprocess_for_position(img)
                kp, des = orb.detectAndCompute(ref_gray, None)
                if des is not None and kp is not None and len(kp) >= POS_MIN_MATCHES:
                    st.pos_ref_gray = ref_gray
                    st.pos_ref_kp = kp
                    st.pos_ref_des = des
                    st.pos_ref_ts = int(time.time())
                    pos_save_ref(stream_label, ref_gray, kp, des)
                    log(f"[pos] stream={stream_label} set ORB reference (kp={len(kp)})")
                    g_pos_ref_age.labels(stream=stream_label).set(0.0)

            if st.pos_ref_gray is not None and st.pos_ref_des is not None and st.pos_ref_kp is not None and reason == 0:
                cur_gray = preprocess_for_position(img)
                changed, dx, dy, rot_deg, scale, mcount, inliers, inlier_ratio = estimate_position_change(
                    orb, st.pos_ref_gray, st.pos_ref_kp, st.pos_ref_des, cur_gray
                )

                g_pos_dx.labels(stream=stream_label).set(dx)
                g_pos_dy.labels(stream=stream_label).set(dy)
                g_pos_rot.labels(stream=stream_label).set(rot_deg)
                g_pos_scale.labels(stream=stream_label).set(scale)
                g_pos_matches.labels(stream=stream_label).set(float(mcount))
                g_pos_inliers.labels(stream=stream_label).set(float(inliers))
                g_pos_inlier_ratio.labels(stream=stream_label).set(float(inlier_ratio))
                if st.pos_ref_ts:
                    g_pos_ref_age.labels(stream=stream_label).set(float(max(0, int(time.time()) - st.pos_ref_ts)))

                if changed:
                    st.pos_changed_consec += 1
                else:
                    st.pos_changed_consec = 0

                if st.pos_changed_consec >= POS_DETECT_MIN_CONSEC:
                    st.pos_changed_until = max(st.pos_changed_until, now + POS_HOLD_SEC)
                    g_pos_changed.labels(stream=stream_label).set(1.0)
                else:
                    if now >= st.pos_changed_until:
                        g_pos_changed.labels(stream=stream_label).set(0.0)

        # ------------------------
        # Two-view logic (zoom-in/zoom-out)
        # - Learn / evaluate only when video OK
        # ------------------------
        if VIEW_ENABLE:
            if reason == 0:
                st.view_good_streak += 1
            else:
                st.view_good_streak = 0
                st.view_unexpected_consec = 0
                if now >= st.view_unexpected_until:
                    g_view_unexpected.labels(stream=stream_label).set(0.0)

            # --- Build fingerprint ---
            edges = view_preprocess_edges(img)
            cur_bits = dhash_bits(edges, VIEW_HASH_W, VIEW_HASH_H)

            # --- Known views ---
            kc = view_known_count(st.view0_bits, st.view1_bits)
            g_view_known.labels(stream=stream_label).set(float(kc))

            swdist = view_switch_distance(st.view0_bits, st.view1_bits)
            g_view_switch_dist.labels(stream=stream_label).set(float(swdist))

            # --- Explicit distances to each learned view ---
            if st.view0_bits is not None:
                d0 = hamming_dist(st.view0_bits, cur_bits)
            else:
                d0 = 999999

            if st.view1_bits is not None:
                d1 = hamming_dist(st.view1_bits, cur_bits)
            else:
                d1 = 999999

            # Export distances explicitly
            g_view_dist0.labels(stream=stream_label).set(float(0 if d0 >= 999999 else d0))
            g_view_dist1.labels(stream=stream_label).set(float(0 if d1 >= 999999 else d1))

            # --- Deterministic best match selection ---
            if d0 <= d1:
                best_idx = 0 if d0 < 999999 else -1
                best_dist = int(d0)
            else:
                best_idx = 1 if d1 < 999999 else -1
                best_dist = int(d1)

            # Old meaning preserved
            g_view_index.labels(stream=stream_label).set(float(best_idx))

            # New meaning: zoom-state on video_view_active
            if zoom_in_view is None or best_idx == -1:
                g_view_active.labels(stream=stream_label).set(-1.0)
            else:
                if int(best_idx) == int(zoom_in_view):
                    g_view_active.labels(stream=stream_label).set(1.0)
                else:
                    g_view_active.labels(stream=stream_label).set(0.0)

            # Distance to closest
            g_view_dist.labels(stream=stream_label).set(float(0 if best_dist >= 999999 else best_dist))

            # --- Learn view0 if none ---
            if (
                reason == 0
                and st.view_good_streak >= VIEW_LEARN_MIN_GOOD_FRAMES
                and st.view0_bits is None
            ):
                st.view0_bits = cur_bits
                view_save_refs(stream_label, st.view0_bits, st.view1_bits)
                log(f"[view] stream={stream_label} learned view0")

                g_view_known.labels(stream=stream_label).set(1.0)
                g_view_switch_dist.labels(stream=stream_label).set(0.0)

            # --- Learn view1 via candidate clustering ---
            if (
                reason == 0
                and st.view_good_streak >= VIEW_LEARN_MIN_GOOD_FRAMES
                and st.view0_bits is not None
                and st.view1_bits is None
            ):
                d0_now = hamming_dist(st.view0_bits, cur_bits)
                if d0_now >= VIEW_LEARN_SECOND_MIN_DIST:
                    st.view2_candidates.append(cur_bits)

                if len(st.view2_candidates) >= VIEW2_CANDIDATE_MIN:
                    v1 = choose_medoid(list(st.view2_candidates))
                    d01 = hamming_dist(st.view0_bits, v1)
                    if d01 >= VIEW_LEARN_SECOND_MIN_DIST:
                        st.view1_bits = v1
                        view_save_refs(stream_label, st.view0_bits, st.view1_bits)
                        st.view2_candidates.clear()
                        log(f"[view] stream={stream_label} learned view1 (dist_from_view0={d01})")

                        g_view_known.labels(stream=stream_label).set(2.0)
                        g_view_switch_dist.labels(stream=stream_label).set(float(d01))

            # --- Unexpected view detection ---
            kc = view_known_count(st.view0_bits, st.view1_bits)
            if kc < 2:
                st.view_unexpected_consec = 0
                if now >= st.view_unexpected_until:
                    g_view_unexpected.labels(stream=stream_label).set(0.0)
            else:
                sw = view_switch_distance(st.view0_bits, st.view1_bits)
                thr = int(max(VIEW_MATCH_MAX_DIST, (sw * VIEW_ALERT_MULT) + VIEW_ALERT_MARGIN))

                unexpected = (best_idx == -1) or (best_dist > thr)

                if reason == 0:
                    if unexpected:
                        st.view_unexpected_consec += 1
                    else:
                        st.view_unexpected_consec = 0

                    if st.view_unexpected_consec >= VIEW_UNEXPECTED_CONSEC:
                        st.view_unexpected_until = max(st.view_unexpected_until, now + VIEW_HOLD_SEC)
                        g_view_unexpected.labels(stream=stream_label).set(1.0)
                        log(
                            f"[view] stream={stream_label} UNEXPECTED_VIEW "
                            f"best_idx={best_idx} best_dist={best_dist} thr={thr} switch_dist={sw} known={kc}"
                        )
                    else:
                        if now >= st.view_unexpected_until:
                            g_view_unexpected.labels(stream=stream_label).set(0.0)

        await asyncio.sleep(sample_interval)

    # Visit clean end
    st.backoff_sec = VISIT_BACKOFF_MIN_SEC
    st.backoff_until = 0.0
    g_monitor_up.labels(stream=stream_label).set(0.0)


# ------------------------
# Worker loop
# ------------------------
async def worker_loop(
    worker_id: int,
    semaphore: asyncio.Semaphore,
    browser_context,
    probe_url: str,
    streams: List[Dict[str, Any]],
    state: Dict[str, StreamState],
    sample_interval: float,
    selectors: List[str],
    orb,
) -> None:
    if not streams:
        raise SystemExit("No streams after sharding")

    idx = worker_id % len(streams)

    while True:
        s = streams[idx]
        stream_id = str(s["id"])
        st = state.setdefault(stream_id, StreamState())
        zoom_in_view = s.get("zoom_in_view", None)

        async with semaphore:
            page = await browser_context.new_page()
            try:
                await monitor_one_visit(
                    page,
                    probe_url,
                    stream_id,
                    st,
                    sample_interval,
                    selectors,
                    orb,
                    zoom_in_view,
                )
            except Exception as e:
                c_visit_errors_total.labels(stream=stream_id).inc()

                now = time.time()
                age = now - (st.last_frame_ts or now)

                g_last_frame_age.labels(stream=stream_id).set(float(age))
                g_monitor_up.labels(stream=stream_id).set(0.0)

                g_ok.labels(stream=stream_id).set(0.0)
                g_bad.labels(stream=stream_id).set(1.0)
                g_reason.labels(stream=stream_id).set(5.0)
                g_stalled.labels(stream=stream_id).set(1.0)

                # do not force unexpected flags on errors; leave last values / hold logic
                log(f"[worker {worker_id}] stream={stream_id} visit error: {type(e).__name__}: {e}")
                try:
                    await debug_dump_page(page, stream_id, tag="visit_error")
                except Exception:
                    pass

                st.backoff_until = now + st.backoff_sec
                st.backoff_sec = min(
                    VISIT_BACKOFF_MAX_SEC,
                    max(VISIT_BACKOFF_MIN_SEC, st.backoff_sec * VISIT_BACKOFF_MULT),
                )
                log(f"[worker {worker_id}] stream={stream_id} backoff_next={st.backoff_sec:.1f}s")

            finally:
                try:
                    await page.close()
                except Exception:
                    pass

        idx = (idx + CONCURRENCY) % len(streams)
        await asyncio.sleep(ROTATE_IDLE_SEC)


# ------------------------
# Main
# ------------------------
async def main_async() -> None:
    probe_url, streams = load_config(STREAMS_CONFIG)
    selectors = parse_selectors()

    start_http_server(METRICS_PORT)

    log(f"[fleet] metrics: http://127.0.0.1:{METRICS_PORT}/metrics")
    log(f"[fleet] config: {STREAMS_CONFIG}")
    log(f"[fleet] probe_url: {probe_url}")
    log(f"[fleet] streams: {len(streams)} (shard {SHARD_INDEX}/{SHARD_TOTAL})")
    log(f"[fleet] mode=rotate concurrency={CONCURRENCY} window={ROTATE_WINDOW_SEC}s fps={SAMPLE_FPS}")
    log(f"[fleet] selectors(preferred first): {selectors}")
    log(f"[fleet] JANUS_WS={JANUS_WS}")
    log(f"[fleet] view_enable={VIEW_ENABLE} pos_enable={POS_ENABLE}")

    if not JANUS_WS:
        log("[fleet] ERROR: JANUS_WS is empty (probe requires ws=...). This will fail.")
        raise SystemExit(2)

    sample_interval = 1.0 / max(0.1, SAMPLE_FPS)
    state: Dict[str, StreamState] = {}

    ensure_dir(DEBUG_DIR)
    if POS_ENABLE:
        ensure_dir(POS_REF_DIR)
    if VIEW_ENABLE:
        ensure_dir(VIEW_REF_DIR)

    orb = cv2.ORB_create(nfeatures=POS_ORB_FEATURES)

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )

        context_kwargs: Dict[str, Any] = {}
        if BASIC_USER and BASIC_PASS:
            context_kwargs["http_credentials"] = {"username": BASIC_USER, "password": BASIC_PASS}

        context = await browser.new_context(**context_kwargs)

        sem = asyncio.Semaphore(CONCURRENCY)

        tasks = []
        for i in range(CONCURRENCY):
            tasks.append(
                asyncio.create_task(
                    worker_loop(i, sem, context, probe_url, streams, state, sample_interval, selectors, orb)
                )
            )

        await asyncio.gather(*tasks)


def main() -> None:
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        log("Interrupted")


if __name__ == "__main__":
    main()
