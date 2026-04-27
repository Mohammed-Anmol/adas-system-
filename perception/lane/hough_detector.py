"""
Lane Departure Warning (LDW) using Hough Line Transform.
Ported from Pi-ADAS logic: detects lane lines via edge detection + Houghlines,
then checks if the vehicle is drifting out of its lane.
"""

import cv2
import numpy as np
import math


class HoughLaneDetector:
    """Detects lane lines using classical CV (Canny + HoughLinesP).
    Best suited for structured roads where lane markings are visible.
    Falls back gracefully when no lines are found.
    """

    def __init__(self, config):
        self.config = config
        self.frame_w = config['camera']['width']
        self.frame_h = config['camera']['height']
        self.deviation_max = config['thresholds']['lane_deviation_max']
        self.cooldown = config['thresholds']['ldw_cooldown_frames']
        self._cooldown_counter = 0

        # Internal state
        self.left_line = None
        self.right_line = None
        self.lane_center = None
        self.departure_side = None  # "left" | "right" | None
        print("[LDW] Hough Lane Detector initialized.")

    # --------------------------------------------------------------------- #
    #  Public API                                                            #
    # --------------------------------------------------------------------- #

    def detect(self, frame):
        """Run full LDW pipeline on a single BGR frame.

        Returns
        -------
        dict with keys:
            left_line   : ((x1,y1),(x2,y2)) or None
            right_line  : ((x1,y1),(x2,y2)) or None
            lane_center : int x-pixel or None
            departure   : "left" | "right" | None
            overlay     : frame with lane lines drawn (BGR)
        """
        if frame is None:
            return self._empty_result(frame)

        h, w = frame.shape[:2]
        self.frame_h, self.frame_w = h, w

        # 1. Pre-process
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        # 2. Region of interest — bottom trapezoid
        mask = self._roi_mask(edges)

        # 3. Hough lines
        lines = cv2.HoughLinesP(mask, rho=1, theta=np.pi / 180,
                                threshold=50, minLineLength=40, maxLineGap=150)

        # 4. Separate left / right by slope
        left_segments, right_segments = self._classify_lines(lines)

        # 5. Average into single left / right line
        self.left_line = self._average_line(left_segments, h)
        self.right_line = self._average_line(right_segments, h)

        # 6. Compute lane centre
        if self.left_line and self.right_line:
            lx = (self.left_line[0][0] + self.left_line[1][0]) / 2
            rx = (self.right_line[0][0] + self.right_line[1][0]) / 2
            self.lane_center = int((lx + rx) / 2)
        else:
            self.lane_center = None

        # 7. Departure check
        self.departure_side = self._check_departure()

        # 8. Draw overlay
        overlay = self._draw_overlay(frame.copy())

        return {
            "left_line": self.left_line,
            "right_line": self.right_line,
            "lane_center": self.lane_center,
            "departure": self.departure_side,
            "overlay": overlay,
        }

    # --------------------------------------------------------------------- #
    #  Internal helpers                                                      #
    # --------------------------------------------------------------------- #

    def _roi_mask(self, edges):
        """Mask out everything except the bottom-centre trapezoid."""
        h, w = edges.shape
        mask = np.zeros_like(edges)
        polygon = np.array([[
            (int(w * 0.05), h),
            (int(w * 0.45), int(h * 0.55)),
            (int(w * 0.55), int(h * 0.55)),
            (int(w * 0.95), h),
        ]], np.int32)
        cv2.fillPoly(mask, polygon, 255)
        return cv2.bitwise_and(edges, mask)

    def _classify_lines(self, lines):
        left, right = [], []
        if lines is None:
            return left, right
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 0.3:
                continue  # nearly horizontal — skip
            if slope < 0:
                left.append((slope, x1, y1, x2, y2))
            else:
                right.append((slope, x1, y1, x2, y2))
        return left, right

    def _average_line(self, segments, h):
        if not segments:
            return None
        slopes, intercepts = [], []
        for slope, x1, y1, x2, y2 in segments:
            intercept = y1 - slope * x1
            slopes.append(slope)
            intercepts.append(intercept)
        avg_slope = np.mean(slopes)
        avg_intercept = np.mean(intercepts)
        if abs(avg_slope) < 1e-6:
            return None
        y1 = h
        y2 = int(h * 0.55)
        x1 = int((y1 - avg_intercept) / avg_slope)
        x2 = int((y2 - avg_intercept) / avg_slope)
        return ((x1, y1), (x2, y2))

    def _check_departure(self):
        if self.lane_center is None:
            self._cooldown_counter = max(0, self._cooldown_counter - 1)
            return None
        vehicle_center = self.frame_w // 2
        offset = (self.lane_center - vehicle_center) / self.frame_w
        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1
            return None
        if offset > self.deviation_max:
            self._cooldown_counter = self.cooldown
            return "right"
        elif offset < -self.deviation_max:
            self._cooldown_counter = self.cooldown
            return "left"
        return None

    def _draw_overlay(self, frame):
        color_left = (255, 0, 0)
        color_right = (0, 0, 255)
        if self.left_line:
            cv2.line(frame, self.left_line[0], self.left_line[1], color_left, 4)
        if self.right_line:
            cv2.line(frame, self.right_line[0], self.right_line[1], color_right, 4)
        if self.lane_center:
            cv2.circle(frame, (self.lane_center, self.frame_h - 30), 8, (0, 255, 0), -1)
        return frame

    def _empty_result(self, frame):
        return {
            "left_line": None, "right_line": None,
            "lane_center": None, "departure": None,
            "overlay": frame,
        }
