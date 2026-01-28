import cv2
import os
import csv
import numpy as np
from keras_facenet import FaceNet
from scipy.spatial.distance import cosine

# ================= CONFIG =================
BASELINE_FRAMES = 20
ANALYZE_EVERY_N_FRAMES = 4
FACENET_EVERY_N_FRAMES = 25
FACENET_THRESHOLD = 0.45

MULTI_FACE_CONFIRM_FRAMES = 5
FACE_MISSING_CONFIRM_FRAMES = 10

LOG_DIR = "data/logs"
SCREENSHOT_DIR = os.path.join(LOG_DIR, "screenshots")
LOG_FILE = os.path.join(LOG_DIR, "offline_proctoring_log.csv")

os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# ================= MODELS =================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
embedder = FaceNet()

# ================= ENGINE =================
def run_engine(shared):
    video_path = shared["video_path"]
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        shared["status"] = "Error: Cannot open video"
        return

    # ---------- SEEK SUPPORT ----------
    seek_time = shared.get("seek_time")
    if seek_time is not None:
        cap.set(cv2.CAP_PROP_POS_MSEC, seek_time * 1000)
        shared["seek_time"] = None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1:
        fps = 25
    frame_delay = int(1000 / fps)

    # ---------- LOGGING ----------
    os.makedirs(LOG_DIR, exist_ok=True)
    log_fp = open(LOG_FILE, "w", newline="")
    logger = csv.writer(log_fp)
    logger.writerow(["Time(s)", "Event", "Evidence"])

    # ---------- STATE ----------
    baseline_embeddings = []
    baseline_embedding = None

    frame_count = 0
    last_faces = []

    # Multiple faces
    multi_face_frames = 0
    multiple_faces_logged = False

    # Face missing
    face_missing_frames = 0
    face_missing_logged = False

    # Replacement
    replacement_logged = False
    baseline_present = True

    shared["violations"].clear()

    # ================= MAIN LOOP =================
    while cap.isOpened() and shared["start"]:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        sec = int(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        display_status = "Normal"

        # ---------- FACE DETECTION (SAMPLED) ----------
        if frame_count % ANALYZE_EVERY_N_FRAMES == 0:
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            last_faces = faces
        else:
            faces = last_faces

        # ---------- MULTIPLE FACES ----------
        if len(faces) > 1:
            multi_face_frames += 1
            if multi_face_frames >= MULTI_FACE_CONFIRM_FRAMES:
                display_status = "Malpractice: Multiple Faces"

                if not multiple_faces_logged:
                    name = f"multiple_faces_{sec}.jpg"
                    cv2.imwrite(os.path.join(SCREENSHOT_DIR, name), frame)

                    shared["violations"].append({
                        "time": sec,
                        "frame": frame_count,
                        "event": display_status,
                        "image": name
                    })

                    logger.writerow([sec, display_status, name])
                    multiple_faces_logged = True
        else:
            multi_face_frames = 0
            multiple_faces_logged = False

        # ---------- FACE MISSING ----------
        if len(faces) == 0:
            face_missing_frames += 1
            display_status = "Suspicious: Face Missing"

            if face_missing_frames >= FACE_MISSING_CONFIRM_FRAMES:
                if not face_missing_logged:
                    name = f"face_missing_{sec}.jpg"
                    cv2.imwrite(os.path.join(SCREENSHOT_DIR, name), frame)

                    shared["violations"].append({
                        "time": sec,
                        "frame": frame_count,
                        "event": display_status,
                        "image": name
                    })

                    logger.writerow([sec, display_status, name])
                    face_missing_logged = True
        else:
            face_missing_frames = 0
            face_missing_logged = False

        # ---------- SINGLE FACE / IDENTITY ----------
        if len(faces) == 1:
            x, y, w, h = faces[0]
            face_img = frame[y:y+h, x:x+w]

            if face_img.size != 0:
                face_img = cv2.resize(face_img, (160, 160))
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

                # Baseline
                if baseline_embedding is None:
                    emb = embedder.embeddings([face_img])[0]
                    baseline_embeddings.append(emb)

                    if len(baseline_embeddings) >= BASELINE_FRAMES:
                        baseline_embedding = np.mean(baseline_embeddings, axis=0)
                        shared["status"] = "Baseline Locked"

                # Identity check
                elif frame_count % FACENET_EVERY_N_FRAMES == 0:
                    emb = embedder.embeddings([face_img])[0]
                    dist = cosine(baseline_embedding, emb)

                    if dist > FACENET_THRESHOLD:
                        baseline_present = False
                        display_status = "Suspicious: Candidate Replacement"

                        if not replacement_logged:
                            name = f"replacement_{sec}.jpg"
                            cv2.imwrite(os.path.join(SCREENSHOT_DIR, name), frame)

                            shared["violations"].append({
                                "time": sec,
                                "frame": frame_count,
                                "event": display_status,
                                "image": name
                            })

                            logger.writerow([sec, display_status, name])
                            replacement_logged = True
                    else:
                        baseline_present = True

        # ---------- VISUALS ----------
        color = (0, 255, 0) if display_status == "Normal" else (0, 0, 255)
        cv2.putText(frame, f"Status: {display_status}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # ---------- SHARE FRAME ----------
        shared["frame"] = frame.copy()
        shared["status"] = display_status

        # ---------- OPENCV WINDOW ----------
        cv2.imshow("Offline Proctoring (OpenCV)", frame)
        if cv2.waitKey(frame_delay) & 0xFF == 27:
            break

    # ================= CLEANUP =================
    cap.release()
    log_fp.close()
    cv2.destroyAllWindows()

    shared["start"] = False
    shared["status"] = "Completed"