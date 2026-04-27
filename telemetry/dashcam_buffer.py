"""
Dashcam ring-buffer that continuously records the last N seconds of
video + telemetry and saves to disk on safety events.
"""
import os
import time
import json
import threading
from collections import deque

import cv2


class DashcamBuffer:
    """Circular buffer storing frames + telemetry; saves on event."""

    def __init__(self, config=None):
        buf_sec = 60
        self.fps = 30
        self.output_dir = "recordings"

        if config:
            buf_sec = config.get('dashcam', {}).get('buffer_seconds', 60)
            self.fps = config.get('camera', {}).get('fps', 30)
            self.output_dir = config.get('dashcam', {}).get(
                'output_dir', 'recordings')

        max_len = buf_sec * self.fps
        self.buffer = deque(maxlen=max_len)
        self._lock = threading.Lock()

        os.makedirs(self.output_dir, exist_ok=True)
        print(f"[Dashcam] Buffer initialized ({buf_sec}s @ {self.fps}fps).")

    def add_frame(self, frame, telemetry=None):
        """Add a frame + telemetry snapshot to the ring buffer."""
        with self._lock:
            self.buffer.append({
                "frame": frame,
                "telemetry": telemetry,
                "timestamp": time.time(),
            })

    def save_buffer(self, event_name="event"):
        """Save the current buffer to a video file + JSON telemetry log."""
        with self._lock:
            frames = list(self.buffer)

        if not frames or frames[0]['frame'] is None:
            print("[Dashcam] No frames to save.")
            return None

        ts = time.strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(self.output_dir, f"{event_name}_{ts}.avi")
        json_path = os.path.join(self.output_dir, f"{event_name}_{ts}.json")

        h, w = frames[0]['frame'].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(video_path, fourcc, self.fps, (w, h))

        telemetry_log = []
        for entry in frames:
            if entry['frame'] is not None:
                writer.write(entry['frame'])
            telemetry_log.append({
                "timestamp": entry['timestamp'],
                "telemetry": entry.get('telemetry'),
            })

        writer.release()

        with open(json_path, 'w') as f:
            json.dump(telemetry_log, f, indent=2, default=str)

        print(f"[Dashcam] Saved: {video_path} ({len(frames)} frames)")
        return video_path
