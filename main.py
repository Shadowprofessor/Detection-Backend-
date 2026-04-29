import asyncio
import io
import json
import time
import os
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, date

import aiosqlite
import cv2
import numpy as np
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from ultralytics import YOLO


# ══════════════════════════════════════════════════════════════
#  STATE — In-memory frame buffer + WebSocket manager
# ══════════════════════════════════════════════════════════════

class FrameBuffer:
    """Thread-safe latest-frame store. Only keeps the newest JPEG."""
    def __init__(self):
        self.data: bytes | None = None
        self.timestamp: float = 0
        self.frame_count: int = 0
        self.fps: float = 0
        self._last_fps_check: float = time.time()
        self._fps_frames: int = 0

    def update(self, jpeg_bytes: bytes):
        now = time.time()
        self.data = jpeg_bytes
        self.timestamp = now
        self.frame_count += 1
        self._fps_frames += 1
        # Recalculate FPS every second
        elapsed = now - self._last_fps_check
        if elapsed >= 1.0:
            self.fps = round(self._fps_frames / elapsed, 1)
            self._fps_frames = 0
            self._last_fps_check = now


class ConnectionManager:
    """WebSocket connection manager for detection data broadcast."""
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, data: dict):
        msg = json.dumps(data)
        dead = []
        for ws in self.active:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


frame_buffer = FrameBuffer()
ws_manager = ConnectionManager()


# ══════════════════════════════════════════════════════════════
#  YOLOv8 DETECTOR
# ══════════════════════════════════════════════════════════════

DETECT_EVERY = int(os.getenv("DETECT_EVERY", "3"))  # YOLO on every 3rd frame
SNAPSHOT_DIR = "snapshots"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

model = YOLO("yolov8n.pt")  # ~6MB, auto-downloads if missing

# COCO classes we care about for surveillance
# Format: class_id -> (name, category, alert_level)
TARGET_CLASSES = {
    # ── People ──
    0:  ("person",       "human",           "high"),
    # ── Vehicles ──
    1:  ("bicycle",      "bicycle",         "low"),
    2:  ("car",          "vehicle",         "low"),
    3:  ("motorcycle",   "vehicle",         "medium"),
    # ── Animals ──
    15: ("cat",          "animal",          "medium"),
    16: ("dog",          "animal",          "medium"),
    17: ("horse",        "animal",          "medium"),
    18: ("sheep",        "animal",          "low"),
    19: ("cow",          "animal",          "medium"),
    20: ("elephant",     "animal",          "high"),
    21: ("bear",         "animal",          "high"),
    # ── Weapons / Dangerous objects ──
    34: ("baseball bat", "weapon",          "critical"),
    43: ("knife",        "weapon",          "critical"),
    76: ("scissors",     "weapon",          "critical"),
    # ── Suspicious items (theft targets / unattended bags) ──
    24: ("backpack",     "suspicious_item", "medium"),
    26: ("handbag",      "suspicious_item", "medium"),
    28: ("suitcase",     "suspicious_item", "medium"),
}

latest_detections = {
    "detections": [], "bicycle_count": 0, "weapon_count": 0,
    "human_alert": False, "animal_alert": False, "weapon_alert": False
}
frame_counter = 0


async def run_detection(jpeg_bytes: bytes, camera_id: str) -> dict:
    """Run YOLOv8n inference in thread pool (non-blocking)."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _detect_sync, jpeg_bytes, camera_id)


def _bbox_proximity(b1: list, b2: list, threshold: float = 0.15) -> bool:
    """Check if two bounding boxes are close / overlapping.
    Uses IoU + edge-distance heuristic — returns True if suspicious proximity."""
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = area1 + area2 - inter
    iou = inter / union if union > 0 else 0
    if iou > threshold:
        return True
    # Also check edge-to-edge distance (within 80px = likely interacting)
    cx1, cy1 = (b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2
    cx2, cy2 = (b2[0] + b2[2]) / 2, (b2[1] + b2[3]) / 2
    dist = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
    avg_size = ((area1 ** 0.5) + (area2 ** 0.5)) / 2
    return dist < avg_size * 1.5  # within 1.5x average object size


def _detect_sync(jpeg_bytes: bytes, camera_id: str) -> dict:
    """Blocking YOLOv8 inference — runs in executor thread."""
    nparr = np.frombuffer(jpeg_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return latest_detections

    results = model(
        frame, conf=0.45, iou=0.5,
        classes=list(TARGET_CLASSES.keys()),
        imgsz=640, verbose=False
    )[0]

    detections = []
    bicycle_count = 0
    weapon_count = 0
    human_alert = False
    animal_alert = False
    weapon_alert = False
    snapshot_path = None
    suspicious = False

    human_bboxes = []
    bicycle_bboxes = []
    animal_bboxes = []
    weapon_bboxes = []
    suspicious_item_bboxes = []

    for box in results.boxes:
        cid = int(box.cls[0])
        if cid not in TARGET_CLASSES:
            continue
        name, category, priority = TARGET_CLASSES[cid]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        bbox = [x1, y1, x2, y2]
        detections.append({
            "class": name, "category": category,
            "confidence": round(conf, 3),
            "bbox": bbox,
            "alert_level": priority
        })

        if category == "bicycle":
            bicycle_count += 1
            bicycle_bboxes.append(bbox)
        elif category == "human":
            human_alert = True
            human_bboxes.append(bbox)
        elif category == "animal":
            animal_alert = True
            animal_bboxes.append(bbox)
        elif category == "weapon":
            weapon_count += 1
            weapon_alert = True
            weapon_bboxes.append(bbox)
        elif category == "suspicious_item":
            suspicious_item_bboxes.append(bbox)

    # ══════════════════════════════════════════════════════════
    #  SUSPICIOUS ACTIVITY DETECTION — priority ordered
    # ══════════════════════════════════════════════════════════
    suspicious_reason = None

    # 1) CRITICAL: Any weapon detected → always suspicious
    if weapon_bboxes:
        suspicious = True
        suspicious_reason = "weapon_detected"

    # 2) CRITICAL: Person holding/near weapon
    if not suspicious:
        for hb in human_bboxes:
            for wb in weapon_bboxes:
                if _bbox_proximity(hb, wb):
                    suspicious = True
                    suspicious_reason = "person_with_weapon"
                    break
            if suspicious:
                break

    # 3) HIGH: Person near bicycle → potential theft
    if not suspicious:
        for hb in human_bboxes:
            for bb in bicycle_bboxes:
                if _bbox_proximity(hb, bb):
                    suspicious = True
                    suspicious_reason = "person_near_bicycle"
                    break
            if suspicious:
                break

    # 4) MEDIUM: Person near suspicious item (grabbing bag/backpack)
    if not suspicious:
        for hb in human_bboxes:
            for sb in suspicious_item_bboxes:
                if _bbox_proximity(hb, sb):
                    suspicious = True
                    suspicious_reason = "person_with_suspicious_item"
                    break
            if suspicious:
                break

    # 5) MEDIUM: Person near animal → unusual interaction
    if not suspicious:
        for hb in human_bboxes:
            for ab in animal_bboxes:
                if _bbox_proximity(hb, ab):
                    suspicious = True
                    suspicious_reason = "person_near_animal"
                    break
            if suspicious:
                break

    # 6) LOW: Multiple categories co-present
    if not suspicious:
        categories_present = sum([
            len(human_bboxes) > 0,
            len(bicycle_bboxes) > 0,
            len(animal_bboxes) > 0,
            len(weapon_bboxes) > 0,
            len(suspicious_item_bboxes) > 0
        ])
        if categories_present >= 2:
            suspicious = True
            suspicious_reason = "multi_category_scene"

    # Save annotated snapshot only on suspicious activity
    if suspicious:
        annotated = results.plot()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        path = f"{SNAPSHOT_DIR}/{ts}.jpg"
        cv2.imwrite(path, annotated)
        snapshot_path = f"/snapshots/{ts}.jpg"

    return {
        "detections": detections,
        "bicycle_count": bicycle_count,
        "weapon_count": weapon_count,
        "human_alert": human_alert,
        "animal_alert": animal_alert,
        "weapon_alert": weapon_alert,
        "suspicious": suspicious,
        "suspicious_reason": suspicious_reason,
        "snapshot_path": snapshot_path,
        "camera_id": camera_id,
        "ts": datetime.utcnow().isoformat() + "Z"
    }


# ══════════════════════════════════════════════════════════════
#  FASTAPI APP
# ══════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize SQLite database on startup."""
    async with aiosqlite.connect("detections.db") as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT,
                camera_id TEXT,
                event_type TEXT,
                detections TEXT,
                bicycle_count INTEGER DEFAULT 0,
                snapshot_path TEXT,
                confidence REAL
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS daily (
                day TEXT PRIMARY KEY,
                humans INTEGER DEFAULT 0,
                animals INTEGER DEFAULT 0,
                bicycles INTEGER DEFAULT 0
            )
        """)
        await db.commit()
    yield


app = FastAPI(
    title="ESP32-CAM MJPEG Relay",
    description="MJPEG video relay + YOLOv8 detection backend",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve saved detection snapshots
app.mount("/snapshots", StaticFiles(directory="snapshots"), name="snapshots")

# Pending camera commands (zoom, etc.) — relayed via polling
pending_commands: dict = {}


# ══════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════

@app.post("/push_frame")
async def push_frame(request: Request):
    """
    Receive JPEG frame from ESP32-CAM at 15fps.
    Store in frame_buffer. Run YOLO every DETECT_EVERY frames.
    Broadcast detection results via WebSocket.
    """
    global frame_counter, latest_detections

    jpeg_bytes = await request.body()
    if not jpeg_bytes:
        return JSONResponse({"error": "empty frame"}, status_code=400)

    camera_id = request.headers.get("X-Camera-ID", "unknown")
    frame_buffer.update(jpeg_bytes)
    frame_counter += 1

    # Run detection asynchronously every Nth frame
    if frame_counter % DETECT_EVERY == 0:
        result = await run_detection(jpeg_bytes, camera_id)
        latest_detections = result

        # Count current objects in THIS frame
        human_count = sum(1 for d in result["detections"] if d["category"] == "human")
        animal_count = sum(1 for d in result["detections"] if d["category"] == "animal")
        bicycle_count = result["bicycle_count"]
        weapon_count = result["weapon_count"]
        vehicle_count = sum(1 for d in result["detections"] if d["category"] == "vehicle")
        suspicious_item_count = sum(1 for d in result["detections"] if d["category"] == "suspicious_item")

        # ALWAYS broadcast detection JSON — even when empty
        # This lets the frontend reset counters to 0 when nothing is seen
        await ws_manager.broadcast({
            "type": "detection",
            **result,
            "human_count": human_count,
            "animal_count": animal_count,
            "bicycle_count": bicycle_count,
            "weapon_count": weapon_count,
            "vehicle_count": vehicle_count,
            "suspicious_item_count": suspicious_item_count,
            "fps": frame_buffer.fps,
            "frame_count": frame_buffer.frame_count
        })

        # Persist to SQLite only if something was detected
        if result["detections"]:
            async with aiosqlite.connect("detections.db") as db:
                event_type = "multi" if sum([
                    result["human_alert"],
                    result["animal_alert"],
                    result["bicycle_count"] > 0
                ]) > 1 else (
                    "human" if result["human_alert"]
                    else "animal" if result["animal_alert"]
                    else "bicycle"
                )
                await db.execute(
                    """INSERT INTO events
                       (ts, camera_id, event_type, detections, bicycle_count, snapshot_path)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (result["ts"], camera_id, event_type,
                     json.dumps(result["detections"]),
                     result["bicycle_count"],
                     result.get("snapshot_path"))
                )
                today = date.today().isoformat()
                await db.execute("""
                    INSERT INTO daily (day, humans, animals, bicycles)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(day) DO UPDATE SET
                      humans   = humans   + excluded.humans,
                      animals  = animals  + excluded.animals,
                      bicycles = bicycles + excluded.bicycles
                """, (
                    today,
                    1 if result["human_alert"] else 0,
                    1 if result["animal_alert"] else 0,
                    result["bicycle_count"]
                ))
                await db.commit()

    return JSONResponse({"ok": True, "fps": frame_buffer.fps})


# ── MJPEG Stream ──────────────────────────────────────────────

async def mjpeg_generator():
    """
    Yield MJPEG multipart frames from the in-memory buffer.
    Browser <img src="/stream"> receives this natively.

    Format: multipart/x-mixed-replace; boundary=frame
    Each part: Content-Type: image/jpeg + JPEG bytes.

    Polls at ~50fps max, yields whenever a new frame arrives.
    Effective rate matches ESP32 push rate (~15fps).
    """
    last_ts = 0.0
    while True:
        if frame_buffer.data and frame_buffer.timestamp > last_ts:
            last_ts = frame_buffer.timestamp
            frame = frame_buffer.data
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n"
                + frame + b"\r\n"
            )
        else:
            await asyncio.sleep(0.02)  # 20ms — no busy-loop


@app.get("/stream")
async def stream():
    """
    MJPEG stream endpoint.
    Usage: <img src="https://backend.railway.app/stream">
    Works natively in Chrome, Firefox, Safari — zero JS needed.
    """
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Access-Control-Allow-Origin": "*",
        }
    )


# ── WebSocket ─────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """
    WebSocket for detection JSON data (NOT video frames).
    Sends: detection events, alerts, counts, fps stats.
    Client sends "ping" → server replies "pong" for keepalive.
    """
    await ws_manager.connect(ws)
    # Send current state immediately on connect
    await ws.send_json({
        "type": "init",
        "fps": frame_buffer.fps,
        "latest": latest_detections
    })
    try:
        while True:
            data = await ws.receive_text()
            if data == "ping":
                await ws.send_text("pong")
    except WebSocketDisconnect:
        ws_manager.disconnect(ws)


# ── Data APIs ─────────────────────────────────────────────────

@app.get("/events")
async def get_events(limit: int = 50, type: str = "all"):
    """Get recent detection events, optionally filtered by type."""
    async with aiosqlite.connect("detections.db") as db:
        db.row_factory = aiosqlite.Row
        if type == "all":
            rows = await db.execute_fetchall(
                "SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,))
        else:
            rows = await db.execute_fetchall(
                "SELECT * FROM events WHERE event_type=? ORDER BY id DESC LIMIT ?",
                (type, limit))
    return [dict(r) for r in rows]


@app.get("/stats")
async def stats():
    """Get today's detection stats and system info."""
    today = date.today().isoformat()
    async with aiosqlite.connect("detections.db") as db:
        db.row_factory = aiosqlite.Row
        row = await db.execute_fetchall(
            "SELECT * FROM daily WHERE day=?", (today,))
    counts = dict(row[0]) if row else {"humans": 0, "animals": 0, "bicycles": 0}
    return {
        **counts,
        "fps": frame_buffer.fps,
        "total_frames": frame_buffer.frame_count
    }


# ── Camera Control Relay ──────────────────────────────────────

@app.post("/cmd/zoom")
async def cmd_zoom(level: int, camera_id: str = "ESP32_CAM_01"):
    """Queue a zoom command for the ESP32 to pick up via polling."""
    pending_commands[camera_id] = {"zoom": level}
    return {"ok": True, "zoom": level}


@app.post("/cmd/af")
async def cmd_af(camera_id: str = "ESP32_CAM_01"):
    """Queue an autofocus command."""
    pending_commands.setdefault(camera_id, {})["af"] = True
    return {"ok": True, "af": "queued"}


@app.get("/cmd/poll")
async def cmd_poll(camera_id: str = "ESP32_CAM_01"):
    """ESP32 polls this to get pending commands. Returns {} if none."""
    cmd = pending_commands.pop(camera_id, {})
    return cmd


# ── Health ────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "ok": True,
        "fps": frame_buffer.fps,
        "total_frames": frame_buffer.frame_count,
        "model": "yolov8n",
        "detect_every": DETECT_EVERY,
        "ws_clients": len(ws_manager.active)
    }


# ══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000")),
        workers=1
    )
