"""
Multi-object tracker using a simplified DeepSORT-style approach.
Combines IoU-based association with a Kalman filter per track.
"""
import numpy as np
from collections import defaultdict


class KalmanBoxTracker:
    """Per-object Kalman filter tracking [cx, cy, area, aspect, vx, vy, va]."""
    _count = 0

    def __init__(self, bbox):
        self.id = KalmanBoxTracker._count
        KalmanBoxTracker._count += 1
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        self.state = np.array([cx, cy, w * h, w / max(h, 1), 0, 0, 0], dtype=float)
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.history = []

    def predict(self):
        self.state[:3] += self.state[4:7]
        self.age += 1
        self.time_since_update += 1
        return self._to_bbox()

    def update(self, bbox):
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        new = np.array([cx, cy, w * h, w / max(h, 1)])
        self.state[4:7] = new[:3] - self.state[:3]
        self.state[:4] = new
        self.time_since_update = 0
        self.hits += 1

    def _to_bbox(self):
        w = np.sqrt(max(self.state[2] * self.state[3], 1))
        h = max(self.state[2] / max(w, 1), 1)
        cx, cy = self.state[0], self.state[1]
        return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def _iou(bb1, bb2):
    x1 = max(bb1[0], bb2[0])
    y1 = max(bb1[1], bb2[1])
    x2 = min(bb1[2], bb2[2])
    y2 = min(bb1[3], bb2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = max((bb1[2] - bb1[0]) * (bb1[3] - bb1[1]), 1e-6)
    a2 = max((bb2[2] - bb2[0]) * (bb2[3] - bb2[1]), 1e-6)
    return inter / (a1 + a2 - inter)


class MultiObjectTracker:
    """Simplified DeepSORT tracker using IoU association + Kalman prediction."""

    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        print("[Tracker] Multi-Object Tracker initialized.")

    def update(self, detections, frame=None):
        """Update tracks with new detections.

        Parameters
        ----------
        detections : list of dicts {x1, y1, x2, y2, confidence, class_name}

        Returns
        -------
        list of dicts {track_id, x1, y1, x2, y2, class_name, velocity}
        """
        # Predict existing tracks
        for t in self.trackers:
            t.predict()

        # Build cost matrix (negative IoU)
        det_bboxes = [[d['x1'], d['y1'], d['x2'], d['y2']] for d in detections]
        trk_bboxes = [t._to_bbox() for t in self.trackers]

        matched, unmatched_dets, unmatched_trks = self._associate(
            det_bboxes, trk_bboxes)

        # Update matched trackers
        for d_idx, t_idx in matched:
            self.trackers[t_idx].update(det_bboxes[d_idx])
            self.trackers[t_idx].class_name = detections[d_idx].get(
                'class_name', 'unknown')

        # Create new trackers for unmatched detections
        for d_idx in unmatched_dets:
            t = KalmanBoxTracker(det_bboxes[d_idx])
            t.class_name = detections[d_idx].get('class_name', 'unknown')
            self.trackers.append(t)

        # Remove dead trackers
        self.trackers = [t for t in self.trackers
                         if t.time_since_update <= self.max_age]

        # Build output
        results = []
        for t in self.trackers:
            if t.hits >= self.min_hits or t.time_since_update == 0:
                bb = t._to_bbox()
                results.append({
                    'track_id': t.id,
                    'x1': bb[0], 'y1': bb[1], 'x2': bb[2], 'y2': bb[3],
                    'class_name': getattr(t, 'class_name', 'unknown'),
                    'velocity': (t.state[4], t.state[5]),
                })
        return results

    def _associate(self, dets, trks):
        if not dets or not trks:
            return [], list(range(len(dets))), list(range(len(trks)))

        iou_matrix = np.zeros((len(dets), len(trks)))
        for d in range(len(dets)):
            for t in range(len(trks)):
                iou_matrix[d, t] = _iou(dets[d], trks[t])

        matched, unmatched_d, unmatched_t = [], [], []
        used_d, used_t = set(), set()

        # Greedy matching
        while True:
            if iou_matrix.size == 0:
                break
            best = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            if iou_matrix[best] < self.iou_threshold:
                break
            d_i, t_i = best
            matched.append((d_i, t_i))
            used_d.add(d_i)
            used_t.add(t_i)
            iou_matrix[d_i, :] = 0
            iou_matrix[:, t_i] = 0

        unmatched_d = [i for i in range(len(dets)) if i not in used_d]
        unmatched_t = [i for i in range(len(trks)) if i not in used_t]
        return matched, unmatched_d, unmatched_t
