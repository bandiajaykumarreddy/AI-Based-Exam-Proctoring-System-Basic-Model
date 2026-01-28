import os
import cv2
import threading
import time
from flask import Flask, render_template, request, Response, jsonify
from offline_video_engine import run_engine

# ================= CONFIG =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ================= SHARED STATE =================
shared = {
    "frame": None,
    "status": "Idle",
    "violations": [],
    "video_path": None,
    "start": False,
    "running": False
}

# ================= FLASK APP =================
app = Flask(
    __name__,
    template_folder="web/templates",
    static_folder="web/static"
)

@app.route("/")
def dashboard():
    return render_template("dashboard.html")

# ---------- Upload Video ----------
@app.route("/upload", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return {"error": "No file"}, 400

    file = request.files["video"]
    if file.filename == "":
        return {"error": "Empty filename"}, 400

    save_path = os.path.join(UPLOAD_DIR, file.filename)
    file.save(save_path)

    shared["video_path"] = save_path
    shared["violations"].clear()
    shared["status"] = "Uploaded"

    return {"status": "uploaded", "path": save_path}

# ---------- Start Analysis ----------
@app.route("/analyze", methods=["POST"])
def analyze():
    if not shared["video_path"]:
        return {"error": "No video uploaded"}, 400

    if shared["running"]:
        return {"status": "already running"}

    shared["start"] = True
    return {"status": "analysis started"}

# ---------- Video Stream to Web ----------
def generate_stream():
    while True:
        if shared["frame"] is None:
            time.sleep(0.05)
            continue

        _, buffer = cv2.imencode(".jpg", shared["frame"])
        frame_bytes = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               frame_bytes + b"\r\n")

@app.route("/video_feed")
def video_feed():
    return Response(
        generate_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

# ---------- Violations API ----------
@app.route("/violations")
def get_violations():
    return jsonify({
        "status": shared["status"],
        "violations": shared["violations"]
    })

# ================= ENGINE THREAD =================
def engine_loop():
    """
    OpenCV MUST run independently of Flask requests.
    This loop waits for 'start' signal from web UI.
    """
    while True:
        if not shared["start"]:
            time.sleep(0.1)
            continue

        shared["running"] = True
        shared["status"] = "Analyzing"

        run_engine(shared)   # <-- OpenCV window opens here

        shared["running"] = False
        shared["start"] = False
        shared["status"] = "Completed"

# ================= MAIN =================
if __name__ == "__main__":
    threading.Thread(target=engine_loop, daemon=True).start()
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)