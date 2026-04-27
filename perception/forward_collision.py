"""
Forward Collision Warning (FCW) and Forward Vehicle Start Alarm (FVSA).
Ported from Pi-ADAS logic.
- FCW: computes Time-To-Collision from bounding box growth rate.
- FVSA: alerts when a stationary lead vehicle starts moving.
"""
import time
import numpy as np


class ForwardCollisionWarning:
    """FCW using bounding-box height growth to approximate TTC.
    FVSA tracks a lead vehicle that was stationary and starts moving."""

    def __init__(self, config):
        thresholds = config['thresholds']
        self.ttc_warn = thresholds.get('ttc_warning', 3.0)
        self.ttc_brake = thresholds.get('ttc_brake', 1.5)
        self.min_box_h = thresholds.get('fcw_min_box_height', 80)
        self.fvsa_stat_frames = thresholds.get('fvsa_stationary_frames', 90)
        self.fvsa_move_thresh = thresholds.get('fvsa_move_threshold', 15)

        # State
        self._prev_heights = {}   # track_id -> list of recent heights
        self._prev_centers = {}   # track_id -> list of recent cx values
        self._stationary_count = {}  # track_id -> frames stationary
        self._last_ts = time.time()

        self.fcw_active = False
        self.emergency_brake = False
        self.fvsa_active = False
        print("[FCW/FVSA] Forward Collision module initialized.")

    def update(self, tracked_objects, ego_speed_kmh=0.0):
        """Process tracked objects for collision and start-alarm.

        Parameters
        ----------
        tracked_objects : list of dicts
            {track_id, x1, y1, x2, y2, class_name}
        ego_speed_kmh : float — vehicle speed from OBD

        Returns
        -------
        dict with keys:
            fcw_warning : bool
            emergency_brake : bool
            fvsa_alert : bool
            ttc : float or None (seconds)
            lead_vehicle_id : int or None
        """
        now = time.time()
        dt = max(now - self._last_ts, 0.001)
        self._last_ts = now

        self.fcw_active = False
        self.emergency_brake = False
        self.fvsa_active = False
        min_ttc = None
        lead_id = None

        for obj in tracked_objects:
            tid = obj.get('track_id')
            cls = obj.get('class_name', '')
            if cls not in ('car', 'truck', 'bus', 'auto', 'bike'):
                continue

            bh = obj['y2'] - obj['y1']
            cx = (obj['x1'] + obj['x2']) / 2

            # --- FCW: TTC from bbox height growth rate ---
            hist = self._prev_heights.get(tid, [])
            hist.append(bh)
            if len(hist) > 10:
                hist = hist[-10:]
            self._prev_heights[tid] = hist

            if len(hist) >= 3 and bh > self.min_box_h:
                growth = (hist[-1] - hist[-3]) / (3 * dt)
                if growth > 0:
                    ttc = bh / growth
                    if min_ttc is None or ttc < min_ttc:
                        min_ttc = ttc
                        lead_id = tid
                    if ttc < self.ttc_brake:
                        self.emergency_brake = True
                        self.fcw_active = True
                    elif ttc < self.ttc_warn:
                        self.fcw_active = True

            # --- FVSA: detect stationary→moving ---
            chist = self._prev_centers.get(tid, [])
            chist.append(cx)
            if len(chist) > self.fvsa_stat_frames + 10:
                chist = chist[-(self.fvsa_stat_frames + 10):]
            self._prev_centers[tid] = chist

            if len(chist) >= self.fvsa_stat_frames:
                window = chist[-self.fvsa_stat_frames:-1]
                spread = max(window) - min(window)
                if spread < self.fvsa_move_thresh:
                    # Was stationary
                    recent_move = abs(chist[-1] - chist[-2])
                    if recent_move > self.fvsa_move_thresh:
                        self.fvsa_active = True

        return {
            "fcw_warning": self.fcw_active,
            "emergency_brake": self.emergency_brake,
            "fvsa_alert": self.fvsa_active,
            "ttc": round(min_ttc, 2) if min_ttc else None,
            "lead_vehicle_id": lead_id,
        }
