"""
ADAS System — Main Integration Pipeline
========================================
Multi-threaded main loop coordinating:
  - Camera capture
  - Multi-model perception (YOLO26, DeepLabv3+, MiDaS, U-Net, Hough)
  - Object tracking (DeepSORT-style)
  - Forward Collision Warning + Forward Vehicle Start Alarm
  - Red Light Violation Warning
  - Lane Departure Warning
  - Blind Spot Monitoring
  - Driver Drowsiness Monitoring
  - Sensor Fusion (EKF)
  - Path Planning (A*)
  - Control (PID + Stanley)
  - Arduino serial actuation
  - Dashcam ring-buffer recording
  - Emergency SMS alerts
  - Pygame HUD display

Architecture: Raspberry Pi 5 + Hailo HAT+ (vision) ↔ Arduino (actuation)
"""

import os
import sys
import time
import yaml
import cv2
import numpy as np

# ---- Module imports ----
from hardware.serial_link import SerialLink
from perception.vision_pipeline import VisionPipeline
from perception.traffic_lights import TrafficLightDetector
from perception.forward_collision import ForwardCollisionWarning
from perception.driver_monitor import DriverMonitor
from perception.blind_spot import BlindSpotMonitor
from tracking.tracker import MultiObjectTracker
from fusion.ekf_fusion import SensorFusionEKF
from planning.path_planner import PathPlanner
from control.controller_logic import ControllerLogic
from safety.emergency_alerts import EmergencyAlerter
from telemetry.dashcam_buffer import DashcamBuffer
from ui.display import AdasDisplay


def load_config(path="config/adas_config.yaml"):
    # Make path relative to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, path)
    with open(full_path, 'r') as f:
        return yaml.safe_load(f)


def open_camera(config):
    """Open the camera or video file specified in config."""
    source = config['camera'].get('source', 0)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[Main] ERROR: Cannot open camera/video source: {source}")
        return None
    w = config['camera'].get('width', 1280)
    h = config['camera'].get('height', 720)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    print(f"[Main] Camera opened: {source} ({w}x{h})")
    return cap


def main():
    # Set CWD to the directory of this script
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    print("=" * 60)
    print("  ADAS SYSTEM — Starting Up")
    print("=" * 60)

    config = load_config()

    # ---- Hardware ----
    arduino = SerialLink(
        port=config['hardware']['arduino_port'],
        baudrate=config['hardware']['arduino_baudrate'],
    )

    # ---- Perception ----
    vision = VisionPipeline(config)
    tl_detector = TrafficLightDetector(config)
    fcw = ForwardCollisionWarning(config)
    driver_mon = DriverMonitor(config)
    blind_spot = BlindSpotMonitor(config)

    # ---- Tracking ----
    tracker = MultiObjectTracker()

    # ---- Fusion / Planning / Control ----
    fusion = SensorFusionEKF()
    planner = PathPlanner(config)
    controller = ControllerLogic(config)

    # ---- Safety / Telemetry ----
    alerter = EmergencyAlerter(config)
    dashcam = DashcamBuffer(config)

    # ---- UI ----
    display = AdasDisplay(config)

    # ---- 0. SOURCE SELECTION MENU ----
    source = display.render_menu()
    if source == "QUIT":
        return
    if source is not None and source != "video":
        config['camera']['source'] = source
    elif source == "video":
        # In a real app, use file dialog. For now, check config or use default.
        print("[Main] Video mode selected. Using source from config.")

    # ---- Camera ----
    cap = open_camera(config)

    # ---- Safety state ----
    sensor_mismatch_count = 0
    mismatch_tol = config['thresholds'].get('sensor_mismatch_tolerance', 3)

    print("[Main] All modules initialized. Entering main loop.")
    print("=" * 60)

    frame_count = 0
    fps = 0.0
    fps_timer = time.time()

    try:
        while True:
            loop_start = time.time()

            # ============================================================
            # 1. CAPTURE
            # ============================================================
            frame = None
            if cap and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    # Loop video or end
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if not ret:
                        print("[Main] Camera feed ended.")
                        break

            # ============================================================
            # 2. SENSOR READ (Arduino ultrasonic / OBD)
            # ============================================================
            sensor_data = arduino.get_sensor_data()

            # ============================================================
            # 3. PERCEPTION (parallel multi-model)
            # ============================================================
            vision_out = vision.process_frame(frame)

            # ============================================================
            # 4. TRACKING
            # ============================================================
            tracked = tracker.update(vision_out['obstacles'], frame)

            # ============================================================
            # 5. TRAFFIC LIGHT CLASSIFICATION (RLVW)
            # ============================================================
            tl_results, red_violation = tl_detector.classify(
                frame, vision_out['traffic_lights'])

            # ============================================================
            # 6. FORWARD COLLISION WARNING + FVSA
            # ============================================================
            fcw_out = fcw.update(tracked)

            # ============================================================
            # 7. LANE DEPARTURE WARNING
            # ============================================================
            lane_out = vision_out.get('lane_hough', {})

            # ============================================================
            # 8. BLIND SPOT MONITORING
            # ============================================================
            bsm_out = blind_spot.update(sensor_data) 

            # ============================================================
            # 9. DRIVER MONITORING
            # ============================================================
            driver_state = driver_mon.analyze_face(frame)

            # ============================================================
            # 10. SENSOR FUSION (EKF)
            # ============================================================
            vision_fusion = {}
            if lane_out.get('lane_center') is not None and frame is not None:
                offset = (lane_out['lane_center'] - frame.shape[1] // 2)
                vision_fusion['heading_offset'] = offset * 0.001

            # Use OBD speed if available
            obd_data = {'speed_kmh': sensor_data.get('obd_speed', 0)}
            ego_state = fusion.update(vision_fusion, None, obd_data)

            # ============================================================
            # 11. PATH PLANNING
            # ============================================================
            path = planner.plan_path(
                vision_out.get('drivable_mask'),
                vision_out['obstacles'],
                ego_state,
            )

            # ============================================================
            # 12. CONTROL
            # ============================================================
            if fcw_out['emergency_brake']:
                commands = controller.emergency_stop()
                display.push_warning("EMERGENCY BRAKE ACTIVATED!")
            elif red_violation:
                commands = controller.emergency_stop()
                display.push_warning("RED LIGHT — STOPPING!")
            else:
                commands = controller.compute_commands(path, ego_state)

            # ============================================================
            # 13. ACTUATION
            # ============================================================
            arduino.send_control(
                commands['steering'], commands['throttle'], commands['brake'])

            # ============================================================
            # 14. SAFETY CHECKS
            # ============================================================
            if driver_state.get('drowsy'):
                display.push_warning("DROWSINESS DETECTED — WAKE UP!")

            if lane_out.get('departure'):
                display.push_warning(f"LANE DEPARTURE: {lane_out['departure'].upper()}")

            # ============================================================
            # 15. UI RENDER
            # ============================================================
            overlay = vision_out.get('overlay', frame)

            # Draw planned path on overlay
            if overlay is not None and len(path) >= 2:
                for i in range(len(path) - 1):
                    pt1 = (int(path[i][0]), int(path[i][1]))
                    pt2 = (int(path[i + 1][0]), int(path[i + 1][1]))
                    cv2.line(overlay, pt1, pt2, (255, 0, 255), 2)

            adas_state = {
                "ego": ego_state,
                "fcw": fcw_out,
                "lane": lane_out,
                "rlvw": {"violation": red_violation, "lights": tl_results},
                "blind_spot": bsm_out,
                "driver": driver_state,
                "tracks": tracked,
                "control": commands,
                "sensors": sensor_data,
                "fps": fps,
                "hardware_ok": arduino.conn is not None,
                "vision_raw": {
                    "depth_map": vision_out.get('depth_map'),
                    "seg_overlay": vision_out.get('seg_overlay'),
                    "lane_unet_overlay": vision_out.get('lane_unet', {}).get('overlay'),
                }
            }

            keep_running = display.render(overlay, adas_state)
            if not keep_running:
                print("[Main] UI closed by user.")
                break

            # ============================================================
            # FPS Tracking
            # ============================================================
            frame_count += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_timer = time.time()

    except KeyboardInterrupt:
        print("\n[Main] Ctrl+C received — shutting down.")
    finally:
        print("[Main] Saving dashcam buffer...")
        dashcam.save_buffer("shutdown")
        if cap:
            cap.release()
        display.cleanup()
        print("[Main] ADAS system stopped.")


if __name__ == "__main__":
    main()
