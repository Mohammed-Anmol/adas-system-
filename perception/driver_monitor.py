"""
Driver drowsiness detection using Eye Aspect Ratio (EAR).
Uses MediaPipe Face Mesh.
"""

import cv2
import numpy as np
import math

# Robust MediaPipe import (works even when mp.solutions is broken)
try:
    import mediapipe as mp
    from mediapipe.python.solutions import face_mesh as mp_face_mesh
    MP_AVAILABLE = True
except Exception:
    MP_AVAILABLE = False


# MediaPipe Face Mesh eye landmark indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]


def _ear(landmarks, eye_indices):
    """Compute Eye Aspect Ratio from 6 landmark points."""
    pts = [landmarks[i] for i in eye_indices]

    # Vertical distances
    v1 = math.dist(pts[1], pts[5])
    v2 = math.dist(pts[2], pts[4])

    # Horizontal distance
    h = math.dist(pts[0], pts[3])

    if h < 1e-6:
        return 0.0  # safer fallback

    return (v1 + v2) / (2.0 * h)


class DriverMonitor:
    """Monitors driver face for drowsiness via EAR."""

    def __init__(self, config=None):
        self.ear_thresh = 0.22
        self.consec_frames = 20

        if config and 'thresholds' in config:
            self.ear_thresh = config['thresholds'].get('ear_threshold', 0.22)
            self.consec_frames = config['thresholds'].get(
                'ear_consec_frames', 20
            )

        self._frame_counter = 0
        self.face_mesh = None

        if MP_AVAILABLE:
            try:
                self.face_mesh = mp_face_mesh.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                print("[DMS] MediaPipe Face Mesh loaded.")
            except Exception as e:
                print(f"[DMS] MediaPipe failed: {e}")
                self.face_mesh = None
        else:
            print("[DMS] MediaPipe unavailable — drowsiness detection disabled.")

    def analyze_face(self, frame):
        """Analyze driver face for drowsiness."""

        if frame is None or self.face_mesh is None:
            return {
                "drowsy": False,
                "ear": 0.0,
                "alert_level": 1.0,
                "eyes_closed_frames": 0
            }

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return {
                "drowsy": False,
                "ear": 0.0,
                "alert_level": 1.0,
                "eyes_closed_frames": self._frame_counter
            }

        face = results.multi_face_landmarks[0]
        h, w = frame.shape[:2]

        # Convert landmarks to pixel coordinates
        landmarks = {
            idx: (lm.x * w, lm.y * h)
            for idx, lm in enumerate(face.landmark)
        }

        # Compute EAR
        left_ear = _ear(landmarks, LEFT_EYE)
        right_ear = _ear(landmarks, RIGHT_EYE)
        avg_ear = (left_ear + right_ear) / 2.0

        # Drowsiness logic
        if avg_ear < self.ear_thresh:
            self._frame_counter += 1
        else:
            self._frame_counter = 0

        drowsy = self._frame_counter >= self.consec_frames

        alert = max(
            0.0,
            min(1.0, 1.0 - (self._frame_counter / self.consec_frames))
        )

        return {
            "drowsy": drowsy,
            "ear": round(avg_ear, 3),
            "alert_level": round(alert, 2),
            "eyes_closed_frames": self._frame_counter,
        }