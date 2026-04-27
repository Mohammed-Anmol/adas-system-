"""
Blind spot monitoring.
Fuses ultrasonic distance readings with rear/side camera detections.
"""


class BlindSpotMonitor:
    """Warns when objects are in the blind spot zone."""

    def __init__(self, config=None):
        self.threshold_cm = 150
        if config and 'thresholds' in config:
            self.threshold_cm = config['thresholds'].get(
                'blind_spot_distance_cm', 150)
        self.left_warning = False
        self.right_warning = False
        print("[BSM] Blind Spot Monitor initialized.")

    def update(self, sensor_data, side_detections=None):
        """Update blind spot state.

        Parameters
        ----------
        sensor_data : str or dict — raw serial from Arduino or parsed dict
        side_detections : list of dicts from side cameras (optional)

        Returns
        -------
        dict: left_warning, right_warning, distances
        """
        self.left_warning = False
        self.right_warning = False
        dist_left = None
        dist_right = None

        # Handle dictionary input (preferred)
        if isinstance(sensor_data, dict):
            dist_left = sensor_data.get('left_dist') or sensor_data.get('left')
            dist_right = sensor_data.get('right_dist') or sensor_data.get('right')
            # Handle possible numeric conversion if values are strings
            try:
                if dist_left is not None and not isinstance(dist_left, (int, float)):
                    dist_left = float(dist_left)
                if dist_right is not None and not isinstance(dist_right, (int, float)):
                    dist_right = float(dist_right)
            except (ValueError, TypeError):
                pass
        
        # Parse ultrasonic data from Arduino string (legacy fallback)
        elif isinstance(sensor_data, str):
            parts = sensor_data.split(',')
            for p in parts:
                p = p.strip()
                if p.startswith('DL:'):
                    try:
                        dist_left = float(p[3:])
                    except ValueError:
                        pass
                elif p.startswith('DR:'):
                    try:
                        dist_right = float(p[3:])
                    except ValueError:
                        pass
                elif p.startswith('D:'):
                    try:
                        dist_right = float(p[2:])
                    except ValueError:
                        pass

        if isinstance(dist_left, (int, float)) and dist_left < self.threshold_cm:
            self.left_warning = True
        if isinstance(dist_right, (int, float)) and dist_right < self.threshold_cm:
            self.right_warning = True

        # Side camera detections — any nearby object triggers warning
        if side_detections:
            for det in side_detections:
                side = det.get('side', 'right')
                if side == 'left':
                    self.left_warning = True
                else:
                    self.right_warning = True

        return {
            "left_warning": self.left_warning,
            "right_warning": self.right_warning,
            "dist_left_cm": dist_left,
            "dist_right_cm": dist_right,
        }
