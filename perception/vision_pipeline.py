"""
Multi-model vision pipeline coordinator.
Runs YOLO26 (object detection), DeepLabv3+ (segmentation), MiDaS (depth),
U-Net lane detection, and Hough lane detection in a unified interface.
Uses ThreadPoolExecutor for parallel inference across models.
"""
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

from hardware.hailo_infer import HailoInference, HAILO_AVAILABLE as HAILO_LIB_AVAILABLE
from perception.depth.estimator import DepthEstimator
from perception.segmentation.segmenter import DrivableAreaSegmenter
from perception.lane.hough_detector import HoughLaneDetector
from perception.lane.unet_detector import UNetLaneDetector


class VisionPipeline:
    """Coordinates multiple perception models with parallel execution."""

    def __init__(self, config):
        self.config = config
        models_cfg = config.get('models', {})
        thresh_cfg = config.get('thresholds', {})
        hw_cfg = config.get('hardware', {})

        # --- YOLO detector ---
        self.yolo = None
        self.hailo = None
        self.yolo_conf = thresh_cfg.get('confidence', 0.5)
        self.yolo_iou = thresh_cfg.get('nms_iou', 0.45)
        
        self.use_hailo = hw_cfg.get('use_hailo', False)

        if self.use_hailo and HAILO_LIB_AVAILABLE:
            hailo_model = models_cfg.get('yolo_hailo_path')
            if hailo_model:
                try:
                    self.hailo = HailoInference(hailo_model)
                    print(f"[Vision] Hailo Accelerator loaded: {hailo_model}")
                except Exception as e:
                    print(f"[Vision] Hailo init failed: {e}")

        if self.hailo is None:
            if YOLO_AVAILABLE:
                yolo_path = models_cfg.get('yolo_path', 'yolov8n.pt')
                try:
                    self.yolo = YOLO(yolo_path)
                    print(f"[Vision] YOLO loaded on CPU/GPU: {yolo_path}")
                except Exception as e:
                    print(f"[Vision] YOLO load failed: {e}")
            else:
                print("[Vision] Ultralytics not installed — YOLO disabled.")

        # --- Sub-modules ---
        self.depth = DepthEstimator(config)
        self.segmenter = DrivableAreaSegmenter(config)
        self.hough_lane = HoughLaneDetector(config)
        self.unet_lane = UNetLaneDetector(config)

        self._executor = ThreadPoolExecutor(max_workers=4)
        print("[Vision] Multi-model pipeline ready.")

    def process_frame(self, frame):
        """Run all perception models on a single frame.

        Returns
        -------
        dict with keys:
            obstacles       : list of dicts {x1,y1,x2,y2,confidence,class_name}
            traffic_lights  : list of dicts (subset of obstacles)
            potholes        : list of dicts (subset of obstacles)
            depth_map       : np.ndarray or None
            drivable_mask   : np.ndarray or None
            seg_overlay     : frame overlay or None
            lane_hough      : dict from HoughLaneDetector
            lane_unet       : dict from UNetLaneDetector
            overlay         : composited frame with all overlays
        """
        if frame is None:
            return self._empty_result(frame)

        # Submit parallel tasks
        futures = {}
        futures['yolo'] = self._executor.submit(self._run_yolo, frame)
        futures['depth'] = self._executor.submit(self.depth.estimate, frame)
        futures['seg'] = self._executor.submit(self.segmenter.segment, frame)
        futures['hough'] = self._executor.submit(self.hough_lane.detect, frame)
        futures['unet'] = self._executor.submit(self.unet_lane.detect, frame)

        results = {}
        for key, fut in futures.items():
            try:
                results[key] = fut.result(timeout=2.0)
            except Exception as e:
                print(f"[Vision] {key} failed: {e}")
                results[key] = None

        # Parse YOLO results
        yolo_out = results.get('yolo') or {
            'obstacles': [], 'traffic_lights': [], 'potholes': []}
        seg_out = results.get('seg') or {"mask": None, "overlay": frame}
        hough_out = results.get('hough') or {
            "left_line": None, "right_line": None, "lane_center": None,
            "departure": None, "overlay": frame}
        unet_out = results.get('unet') or {
            "mask": None, "polylines": [], "overlay": frame}

        # Composite overlay: start with lane overlay, add seg tint
        overlay = frame.copy()
        if hough_out.get('overlay') is not None:
            overlay = hough_out['overlay']

        # Add segmentation tint
        if seg_out.get('mask') is not None:
            green = np.zeros_like(overlay)
            green[:, :, 1] = seg_out['mask']
            overlay = cv2.addWeighted(overlay, 1.0, green, 0.2, 0)

        # Draw YOLO boxes
        for det in yolo_out['obstacles']:
            x1, y1 = int(det['x1']), int(det['y1'])
            x2, y2 = int(det['x2']), int(det['y2'])
            label = f"{det['class_name']} {det['confidence']:.1f}"
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(overlay, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        return {
            "obstacles": yolo_out['obstacles'],
            "traffic_lights": yolo_out['traffic_lights'],
            "potholes": yolo_out['potholes'],
            "depth_map": results.get('depth'),
            "drivable_mask": seg_out.get('mask'),
            "seg_overlay": seg_out.get('overlay'),
            "lane_hough": hough_out,
            "lane_unet": unet_out,
            "overlay": overlay,
        }

    def _run_yolo(self, frame):
        """Run YOLO (via Hailo or Ultralytics) and separate results by class."""
        obstacles = []
        traffic_lights = []
        potholes = []

        if self.hailo:
            # Hailo inference
            hailo_out = self.hailo.infer(frame)
            # Assuming Hailo model returns boxes, classes, scores
            # This part needs adjustment based on specific HEF output mapping
            boxes = hailo_out.get('boxes', [])
            classes = hailo_out.get('classes', [])
            scores = hailo_out.get('scores', [])
            
            for i in range(len(boxes)):
                if scores[i] < self.yolo_conf:
                    continue
                cls_id = int(classes[i])
                cls_name = self.config['models']['yolo_classes'][cls_id] if cls_id < len(self.config['models']['yolo_classes']) else 'unknown'
                det = {
                    'x1': float(boxes[i][0]),
                    'y1': float(boxes[i][1]),
                    'x2': float(boxes[i][2]),
                    'y2': float(boxes[i][3]),
                    'confidence': float(scores[i]),
                    'class_name': cls_name,
                }
                self._sort_detection(det, obstacles, traffic_lights, potholes)
            return {"obstacles": obstacles, "traffic_lights": traffic_lights, "potholes": potholes}

        if self.yolo is None:
            return {"obstacles": obstacles, "traffic_lights": traffic_lights,
                    "potholes": potholes}

        results = self.yolo.predict(frame, conf=self.yolo_conf,
                                     iou=self.yolo_iou, verbose=False)

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = r.names.get(cls_id, 'unknown')
                det = {
                    'x1': float(box.xyxy[0][0]),
                    'y1': float(box.xyxy[0][1]),
                    'x2': float(box.xyxy[0][2]),
                    'y2': float(box.xyxy[0][3]),
                    'confidence': float(box.conf[0]),
                    'class_name': cls_name,
                }
                self._sort_detection(det, obstacles, traffic_lights, potholes)

        return {"obstacles": obstacles, "traffic_lights": traffic_lights,
                "potholes": potholes}

    def _sort_detection(self, det, obstacles, traffic_lights, potholes):
        cls_name = det['class_name']
        if cls_name == 'traffic light':
            traffic_lights.append(det)
        elif cls_name == 'pothole':
            potholes.append(det)
        else:
            obstacles.append(det)

    def _empty_result(self, frame):
        return {
            "obstacles": [], "traffic_lights": [], "potholes": [],
            "depth_map": None, "drivable_mask": None,
            "seg_overlay": frame, "lane_hough": {}, "lane_unet": {},
            "overlay": frame,
        }
