import cv2
import mediapipe as mp
import numpy as np
import time
import os
import threading
import urllib.request

from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

try:
    import pygame
    pygame.mixer.init()
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("[WARNING] pygame not found. Install it with:  pip install pygame")

EAR_THRESHOLD = 0.22
EYE_CLOSED_SECONDS = 2.5

# --- FIX: original path pointed to the author's own machine.
# This now points at the mp3 that ships in this repo, next to this script. ---
ALARM_SOUND_FILE = "wake_alarm.mp3"

LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

# ---------------------------------------------------------------
# Same fix as the sunflower project: mp.solutions.face_mesh is
# missing from recent mediapipe pip builds (0.10.30+), so we use
# the new Tasks API (FaceLandmarker) instead. This downloads a
# small model file once on first run.
# ---------------------------------------------------------------
MODEL_PATH = "face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading face landmark model (first run only)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded.")

base_options = mp_tasks.BaseOptions(model_asset_path=MODEL_PATH)
face_landmarker_options = mp_vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=mp_vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)


def eye_aspect_ratio(landmarks, eye_indices, frame_w, frame_h):
    pts = []
    for idx in eye_indices:
        lm = landmarks[idx]
        pts.append(np.array([lm.x * frame_w, lm.y * frame_h]))
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    ear = (A + B) / (2.0 * C)
    return ear

def play_alarm():
    if not PYGAME_AVAILABLE:
        print("\a[ALARM] Eyes closed too long! (No sound – pygame not installed)")
        return
    sound_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ALARM_SOUND_FILE)
    if not os.path.isfile(sound_path):
        print(f"[ALARM] Sound file '{ALARM_SOUND_FILE}' not found – playing beep only.")
        print("\a")
        return
    try:
        pygame.mixer.music.load(sound_path)
        pygame.mixer.music.play(-1)
    except Exception as e:
        print(f"[ALARM] Could not play sound: {e}")

def stop_alarm():
    if not PYGAME_AVAILABLE:
        return
    try:
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
    except Exception:
        pass

def draw_status_overlay(frame, alarm_active, elapsed, ear_avg):
    h, w = frame.shape[:2]
    if alarm_active:
        color = (0, 0, 220)
        bg_color = (0, 0, 180)
        label = "SLEEPING! WAKE UP!"
    else:
        color = (0, 200, 50)
        bg_color = (0, 150, 40)
        label = "AWAKE"

    if alarm_active:
        pulse = int(abs(np.sin(time.time() * 4)) * 80)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.20 + pulse / 1000, frame, 0.80 - pulse / 1000, 0, frame)

    pill_w, pill_h = 340, 54
    pill_x = (w - pill_w) // 2
    pill_y = 14
    cv2.rectangle(frame, (pill_x, pill_y), (pill_x + pill_w, pill_y + pill_h), bg_color, -1, cv2.LINE_AA)
    cv2.rectangle(frame, (pill_x, pill_y), (pill_x + pill_w, pill_y + pill_h), color, 2, cv2.LINE_AA)
    cv2.putText(frame, label, (pill_x + 20, pill_y + 36), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    dot_x = pill_x - 30
    dot_y = pill_y + pill_h // 2
    cv2.circle(frame, (dot_x, dot_y), 14, color, -1, cv2.LINE_AA)
    cv2.circle(frame, (dot_x, dot_y), 14, (255, 255, 255), 2, cv2.LINE_AA)

    bar_h = 50
    bar_y = h - bar_h
    cv2.rectangle(frame, (0, bar_y), (w, h), (20, 20, 20), -1)
    ear_text = f"EAR: {ear_avg:.3f} (thresh: {EAR_THRESHOLD})"
    closed_text = f"Eyes closed: {elapsed:.1f}s / {EYE_CLOSED_SECONDS}s"
    cv2.putText(frame, ear_text, (14, bar_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, closed_text, (14, bar_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    if elapsed > 0 and not alarm_active:
        progress = min(elapsed / EYE_CLOSED_SECONDS, 1.0)
        bar_fill_w = int((w - 28) * progress)
        cv2.rectangle(frame, (14, bar_y + 45), (14 + bar_fill_w, bar_y + 49), (0, 165, 255), -1, cv2.LINE_AA)

    return frame

def draw_face_box(frame, landmarks, color, w, h):
    xs = [int(lm.x * w) for lm in landmarks]
    ys = [int(lm.y * h) for lm in landmarks]
    pad = 10
    x1, y1 = max(0, min(xs) - pad), max(0, min(ys) - pad)
    x2, y2 = min(w, max(xs) + pad), min(h, max(ys) + pad)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    corner = 18
    thick = 3
    for cx, cy, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(frame, (cx, cy), (cx + dx*corner, cy), color, thick, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx, cy + dy*corner), color, thick, cv2.LINE_AA)

def main():
    face_landmarker = mp_vision.FaceLandmarker.create_from_options(face_landmarker_options)

    # --- FIX: index 1 assumes a second/external camera.
    # Index 0 is the default/built-in webcam on most machines. ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    eye_closed_start = None
    alarm_active = False
    frame_timestamp_ms = 0

    print("=" * 60)
    print(" Sleep Detector — press Q to quit")
    print(f" Alarm fires after {EYE_CLOSED_SECONDS}s of closed eyes")
    if PYGAME_AVAILABLE:
        print(f" Looking for alarm sound: {ALARM_SOUND_FILE}")
    print("=" * 60)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        frame_timestamp_ms += 33  # ~30fps pacing for the VIDEO running mode
        results = face_landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        ear_avg = 1.0
        elapsed = 0.0
        face_detected = False

        if results.face_landmarks:
            face_detected = True
            lms = results.face_landmarks[0]

            ear_l = eye_aspect_ratio(lms, LEFT_EYE_INDICES, w, h)
            ear_r = eye_aspect_ratio(lms, RIGHT_EYE_INDICES, w, h)
            ear_avg = (ear_l + ear_r) / 2.0

            eyes_closed = ear_avg < EAR_THRESHOLD

            if eyes_closed:
                if eye_closed_start is None:
                    eye_closed_start = time.time()
                elapsed = time.time() - eye_closed_start
                if elapsed >= EYE_CLOSED_SECONDS and not alarm_active:
                    alarm_active = True
                    threading.Thread(target=play_alarm, daemon=True).start()
            else:
                eye_closed_start = None
                if alarm_active:
                    alarm_active = False
                    stop_alarm()

            face_color = (0, 0, 220) if alarm_active else (0, 220, 60)
            draw_face_box(frame, lms, face_color, w, h)
        else:
            eye_closed_start = None
            if alarm_active:
                alarm_active = False
                stop_alarm()

        frame = draw_status_overlay(frame, alarm_active, elapsed, ear_avg)

        if not face_detected:
            cv2.putText(frame, "No face detected", (w // 2 - 130, h // 2), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 180, 255), 2, cv2.LINE_AA)

        cv2.imshow("Sleep Alarm Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stop_alarm()
    cap.release()
    face_landmarker.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()