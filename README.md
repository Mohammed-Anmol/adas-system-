# 🏎️ ADAS System - Advanced Digital Cockpit

A comprehensive, real-time Level-2/Level-3 Advanced Driver Assistance System (ADAS) optimized for unstructured environments. Featuring a premium **BMW iDrive-inspired UI** and a modular high-performance vision pipeline.

![Status](https://img.shields.io/badge/Status-Beta-orange)
![Python](https://img.shields.io/badge/Python-3.14+-blue)
![Pygame](https://img.shields.io/badge/UI-Pygame--CE-green)

---

## ✨ Key Features

### 🖥️ BMW iDrive-Style Dashboard
A premium digital cockpit with high-fidelity aesthetics:
*   **Dynamic Speedometer**: Real-time speed visualization with sleek arc animations.
*   **Steering Angle Indicator**: Visual feedback for current vehicle orientation.
*   **Glassmorphic Overlays**: Modern, semi-transparent UI elements for a premium feel.
*   **Intelligent Alerts**: Pulsing warning banners for Forward Collision and Lane Departure.

### 🔳 CP PLUS Style Grid View (Multi-View)
Switch instantly to a professional 4-quadrant monitoring mode:
*   **Vision Feed**: Real-time YOLO object detection.
*   **Depth Heatmap**: 3D spatial awareness via MiDaS.
*   **Drivable Area**: AI-powered road segmentation.
*   **Lane Segmentation**: Precision UNet lane line extraction.

### 🧠 Advanced Perception Pipeline
*   **Object Detection**: YOLOv8-based tracking for vehicles, pedestrians, and traffic lights.
*   **Lane Departure Warning (LDW)**: Dual-mode detection (Hough + UNet).
*   **Forward Collision Warning (FCW)**: Time-to-Collision (TTC) calculations.
*   **Driver Monitoring (DMS)**: Eye-tracking for drowsiness detection.
*   **Sensor Fusion**: Extended Kalman Filter (EKF) combining vision and ultrasonic data.

---

## 🎮 Shortcut Keys

| Key | Action |
| :--- | :--- |
| **`F`** | Toggle **Full Screen** mode |
| **`G`** | Toggle **Grid View** (Multi-Display) |
| **`V`** | Cycle **Focus Layer** (Main ➡️ Depth ➡️ Seg ➡️ Lane) |
| **`ESC`** | Safe Shutdown |

---

## 🚀 Getting Started

### Prerequisites
*   Python 3.10+ (Tested on 3.14)
*   Webcam or Video File
*   Arduino (optional, for sensor data)

### Installation
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Mohammed-Anmol/adas-system-.git
    cd adas-system-
    ```

2.  **Set up Virtual Environment:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the System:**
    ```bash
    python main.py
    ```

---

## 🏗️ Architecture

The system uses a split-architecture for low-latency processing:
*   **Vision Engine**: Multi-threaded model inference (YOLO, MiDaS, UNet).
*   **Logic Engine**: Path planning, control (PID+Stanley), and safety state machine.
*   **UI Engine**: Pygame-CE driven HUD with hardware acceleration.

---

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

---

Developed with ❤️ for Advanced Agentic Coding.
