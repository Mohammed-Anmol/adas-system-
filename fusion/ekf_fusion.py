"""
Extended Kalman Filter (EKF) for sensor fusion.
Fuses vision-based estimates, IMU data, and OBD speed to estimate ego state.
Adapted from PythonRobotics EKF localization.
"""
import numpy as np
import math


class SensorFusionEKF:
    """6-state EKF: [x, y, yaw, speed, yaw_rate, accel]."""

    def __init__(self):
        self.dt = 1.0 / 30.0  # default 30 fps
        # State: [x, y, yaw, v, yaw_rate, a]
        self.x = np.zeros(6)
        # Covariance
        self.P = np.eye(6) * 10.0
        # Process noise
        self.Q = np.diag([0.1, 0.1, 0.01, 0.5, 0.01, 0.3])
        # Measurement noise — OBD speed
        self.R_obd = np.diag([1.0])
        # Measurement noise — IMU (yaw_rate, accel)
        self.R_imu = np.diag([0.05, 0.3])
        # Measurement noise — vision heading offset
        self.R_vis = np.diag([0.2])

        self._initialized = False
        print("[EKF] Sensor Fusion initialized.")

    def predict(self, dt=None):
        """Predict step using constant-turn-rate-acceleration model."""
        if dt is not None:
            self.dt = dt
        dt = self.dt
        x, y, yaw, v, yr, a = self.x

        # State transition
        x_new = x + v * math.cos(yaw) * dt
        y_new = y + v * math.sin(yaw) * dt
        yaw_new = yaw + yr * dt
        v_new = v + a * dt
        yr_new = yr
        a_new = a

        self.x = np.array([x_new, y_new, yaw_new, v_new, yr_new, a_new])

        # Jacobian of state transition
        F = np.eye(6)
        F[0, 2] = -v * math.sin(yaw) * dt
        F[0, 3] = math.cos(yaw) * dt
        F[1, 2] = v * math.cos(yaw) * dt
        F[1, 3] = math.sin(yaw) * dt
        F[2, 4] = dt
        F[3, 5] = dt

        self.P = F @ self.P @ F.T + self.Q

    def update_obd(self, speed_mps):
        """Update with OBD speed measurement (m/s)."""
        H = np.zeros((1, 6))
        H[0, 3] = 1.0
        z = np.array([speed_mps])
        self._kalman_update(z, H, self.R_obd)

    def update_imu(self, yaw_rate, accel):
        """Update with IMU measurements (rad/s, m/s^2)."""
        H = np.zeros((2, 6))
        H[0, 4] = 1.0  # yaw_rate
        H[1, 5] = 1.0  # accel
        z = np.array([yaw_rate, accel])
        self._kalman_update(z, H, self.R_imu)

    def update_vision(self, heading_offset):
        """Update with vision-based heading offset (rad)."""
        H = np.zeros((1, 6))
        H[0, 2] = 1.0
        z = np.array([self.x[2] + heading_offset])
        self._kalman_update(z, H, self.R_vis)

    def _kalman_update(self, z, H, R):
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return
        self.x = self.x + K @ y
        I = np.eye(len(self.x))
        self.P = (I - K @ H) @ self.P

    def update(self, vision_data, imu_data, obd_data):
        """Convenience method: run predict + all available sensor updates.

        Parameters
        ----------
        vision_data : dict or None — expects 'heading_offset' key
        imu_data    : dict or None — expects 'yaw_rate', 'accel' keys
        obd_data    : dict or None — expects 'speed_kmh' key

        Returns
        -------
        dict: ego_x, ego_y, ego_yaw, ego_speed, ego_yaw_rate
        """
        self.predict()

        if obd_data and 'speed_kmh' in obd_data:
            self.update_obd(obd_data['speed_kmh'] / 3.6)

        if imu_data:
            yr = imu_data.get('yaw_rate', 0.0)
            ac = imu_data.get('accel', 0.0)
            self.update_imu(yr, ac)

        if vision_data and 'heading_offset' in vision_data:
            self.update_vision(vision_data['heading_offset'])

        return {
            "ego_x": float(self.x[0]),
            "ego_y": float(self.x[1]),
            "ego_yaw": float(self.x[2]),
            "ego_speed": float(self.x[3]),
            "ego_yaw_rate": float(self.x[4]),
        }
