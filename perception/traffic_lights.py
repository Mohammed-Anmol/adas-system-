"""
Red Light Violation Warning (RLVW) — from Pi-ADAS logic.
Detects traffic lights from YOLO detections, then classifies colour via HSV.
"""
import cv2
import numpy as np


class TrafficLightDetector:
    """Classify detected traffic-light bounding boxes as red/yellow/green."""

    RED_LO1 = np.array([0, 100, 100])
    RED_HI1 = np.array([10, 255, 255])
    RED_LO2 = np.array([160, 100, 100])
    RED_HI2 = np.array([180, 255, 255])
    YEL_LO = np.array([15, 100, 100])
    YEL_HI = np.array([35, 255, 255])
    GRN_LO = np.array([40, 50, 50])
    GRN_HI = np.array([90, 255, 255])

    def __init__(self, config):
        self.conf_thresh = config['thresholds'].get('red_light_confidence', 0.6)
        self.violation = False
        print("[RLVW] Traffic Light Detector initialized.")

    def classify(self, frame, tl_boxes):
        """Classify each traffic-light bbox colour.

        Parameters
        ----------
        frame : BGR image
        tl_boxes : list of dicts {x1, y1, x2, y2, confidence}

        Returns
        -------
        results : list of dicts with added 'colour' key
        violation : bool — True if red light detected
        """
        results = []
        self.violation = False
        if frame is None or not tl_boxes:
            return results, False

        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        for box in tl_boxes:
            x1 = max(0, int(box['x1']))
            y1 = max(0, int(box['y1']))
            x2 = min(w, int(box['x2']))
            y2 = min(h, int(box['y2']))
            if x2 <= x1 or y2 <= y1:
                continue

            roi = hsv[y1:y2, x1:x2]
            colour = self._classify_hsv(roi)
            results.append({**box, 'colour': colour})
            if colour == 'red':
                self.violation = True

        return results, self.violation

    def _classify_hsv(self, roi_hsv):
        r1 = cv2.countNonZero(cv2.inRange(roi_hsv, self.RED_LO1, self.RED_HI1))
        r2 = cv2.countNonZero(cv2.inRange(roi_hsv, self.RED_LO2, self.RED_HI2))
        red = r1 + r2
        yel = cv2.countNonZero(cv2.inRange(roi_hsv, self.YEL_LO, self.YEL_HI))
        grn = cv2.countNonZero(cv2.inRange(roi_hsv, self.GRN_LO, self.GRN_HI))
        counts = {'red': red, 'yellow': yel, 'green': grn}
        best = max(counts, key=counts.get)
        total = roi_hsv.shape[0] * roi_hsv.shape[1]
        if counts[best] < total * 0.05:
            return 'unknown'
        return best
