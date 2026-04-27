# 📋 ADAS System Project Plan

## 1. Project Overview
The objective of this project is to develop a modular, real-time Level-2/Level-3 Advanced Driver Assistance System (ADAS). The system is specifically designed to handle **unstructured environments** (such as Indian roads) where traditional lane markings may be absent or inconsistent.

### Vision
To provide a premium, AI-driven safety layer for vehicles that combines state-of-the-art computer vision with robust hardware integration, offering both safety alerts and semi-autonomous control capabilities.

---

## 2. System Architecture
The system follows a **Modular Pipeline Architecture**, decoupling perception, logic, and visualization for maximum performance and maintainability.

### A. Perception Layer (The "Eyes")
Uses a multi-model ensemble to interpret the surrounding environment:
*   **Object Detection (YOLOv8)**: Identifies and tracks vehicles, pedestrians, and obstacles.
*   **Depth Estimation (MiDaS)**: Generates a dense depth map for spatial awareness without stereo cameras.
*   **Drivable Area Segmentation (DeepLabv3+)**: Distinguishes between road and non-road surfaces.
*   **Lane Detection (Dual-Mode)**: Combines Hough Transforms for structured roads and UNet Segmentation for unstructured paths.

### B. Logic & Planning Layer (The "Brain")
Processes perception data into actionable maneuvers:
*   **Sensor Fusion (EKF)**: Merges vision-based heading data with ultrasonic distance readings and OBD-II telemetry using an Extended Kalman Filter.
*   **Path Planning (A*)**: Calculates an optimal trajectory through the drivable area while avoiding tracked obstacles.
*   **Control Theory**: Uses a hybrid PID and Stanley Controller for smooth steering and throttle/brake actuation.

### C. Visualization Layer (The "Cockpit")
Provides a high-end interface for the driver:
*   **BMW iDrive Style HUD**: A premium focus-mode dashboard.
*   **CP PLUS Grid View**: A multi-display diagnostic grid showing raw perception layers.

---

## 3. Technology Stack

| Category | Technologies |
| :--- | :--- |
| **Language** | Python 3.10+ |
| **Deep Learning** | PyTorch, Torchvision, Ultralytics (YOLOv8) |
| **Computer Vision** | OpenCV, MediaPipe (DMS) |
| **UI/Graphics** | Pygame-CE (Hardware Accelerated) |
| **Hardware Link** | Serial (Arduino), OBD-II (ELM327) |
| **Data Logic** | NumPy, SciPy (EKF) |

---

## 4. Methodology

### Step 1: Data Acquisition
Frames are captured from a high-speed USB/CSI camera. Simultaneously, proximity data is polled from an Arduino-based ultrasonic array via Serial.

### Step 2: Parallel Perception
Models run in a prioritized order:
1.  **Safety Critical**: YOLO (Obstacles) + FCW (Collision Warning).
2.  **Navigation**: Segmentation + Lane Detection.
3.  **Spatial**: MiDaS Depth estimation.

### Step 3: Risk Assessment
The system calculates **Time-to-Collision (TTC)**. If TTC falls below a configurable threshold, the "Emergency Alerter" triggers a haptic or visual warning and initiates an emergency stop command.

### Step 4: Actuation & Feedback
Commands are sent to the vehicle's actuators (or simulated in the UI) while the driver receives real-time telemetry on the BMW-style HUD.

---

## 5. Model Specifications

*   **YOLOv8n**: Chosen for its balance between speed and accuracy on edge devices.
*   **MiDaS v2.1 Small**: Optimized for real-time depth inference on mobile/embedded CPUs.
*   **UNet (Carvana)**: Fine-tuned for precise road boundary segmentation.
*   **DeepLabv3+ (MobileNetV2)**: Used for high-efficiency semantic segmentation of the drivable road area.

---

## 6. Future Roadmap
*   **V2X Integration**: Vehicle-to-Everything communication for traffic light synchronization.
*   **Edge Acceleration**: Optimization for Hailo-8L AI accelerators.
*   **NIGHT-VISION**: Integration of IR camera feeds into the perception pipeline.
*   **Cloud Telemetry**: Real-time dashcam uploads for fleet management.

---

*Document Version: 1.0.0*  
*Last Updated: April 2026*
