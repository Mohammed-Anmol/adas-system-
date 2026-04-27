"""
PID + lateral Stanley controller for steering, throttle, and brake.
Converts a planned path into actuator commands sent to Arduino.
"""
import math
import numpy as np


class PIDController:
    """Simple PID for speed control."""

    def __init__(self, kp=1.0, ki=0.05, kd=0.1, limit=100):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.limit = limit
        self._integral = 0.0
        self._prev_error = 0.0

    def compute(self, error, dt):
        self._integral += error * dt
        self._integral = np.clip(self._integral, -self.limit, self.limit)
        derivative = (error - self._prev_error) / max(dt, 1e-6)
        self._prev_error = error
        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        return np.clip(output, -self.limit, self.limit)


class ControllerLogic:
    """Generates steering / throttle / brake from planned path + ego state."""

    def __init__(self, config=None):
        self.speed_pid = PIDController(kp=2.0, ki=0.1, kd=0.3, limit=100)
        self.target_speed = 30.0  # km/h default cruise
        self.wheelbase = 2.5     # metres (approx for small car)
        self.max_steer_deg = 35
        self.dt = 1.0 / 30.0
        print("[Control] PID + Stanley Controller initialized.")

    def compute_commands(self, planned_path, ego_state):
        """Compute steering, throttle, brake.

        Parameters
        ----------
        planned_path : list of (x, y) pixel waypoints
        ego_state    : dict {ego_speed, ego_yaw, ...}

        Returns
        -------
        dict: steering (0-180, 90=center), throttle (0-255), brake (0-255)
        """
        speed = 0.0
        if ego_state:
            speed = ego_state.get('ego_speed', 0.0)

        # --- Steering via Stanley ---
        steer_deg = 0.0
        if planned_path and len(planned_path) >= 2:
            steer_deg = self._stanley_steering(planned_path, speed)

        # Map from degrees to servo value (90 = straight)
        steer_servo = int(np.clip(90 + steer_deg, 90 - self.max_steer_deg,
                                  90 + self.max_steer_deg))

        # --- Speed PID ---
        speed_kmh = speed * 3.6
        speed_error = self.target_speed - speed_kmh
        pid_out = self.speed_pid.compute(speed_error, self.dt)

        throttle = int(np.clip(pid_out, 0, 255)) if pid_out > 0 else 0
        brake = int(np.clip(-pid_out, 0, 255)) if pid_out < 0 else 0

        return {
            "steering": steer_servo,
            "throttle": throttle,
            "brake": brake,
            "target_speed_kmh": self.target_speed,
            "current_speed_kmh": round(speed_kmh, 1),
        }

    def _stanley_steering(self, path, speed):
        """Simplified Stanley lateral controller.
        Returns steering angle in degrees."""
        # Look-ahead: pick point ~1/3 into the path
        idx = min(len(path) // 3, len(path) - 1)
        idx = max(idx, 1)
        tx, ty = path[idx]

        # Assume ego is at bottom center of frame
        ex, ey = path[0]

        dx = tx - ex
        dy = -(ty - ey)  # flip y (image coords)

        heading_error = math.atan2(dx, max(dy, 1))
        heading_deg = math.degrees(heading_error)

        # Cross-track gain
        k = 0.5
        cross_track = dx
        v = max(abs(speed), 0.5)
        cross_term = math.degrees(math.atan2(k * cross_track, v))

        steer = heading_deg + cross_term
        return np.clip(steer, -self.max_steer_deg, self.max_steer_deg)

    def emergency_stop(self):
        """Return full-brake command."""
        return {"steering": 90, "throttle": 0, "brake": 255}
