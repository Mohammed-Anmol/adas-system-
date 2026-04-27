import serial
import time

class SerialLink:
    def __init__(self, port='/dev/ttyUSB0', baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.conn = None
        self._connect()

    def _connect(self):
        try:
            self.conn = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2) # Wait for arduino to reset
            print(f"Connected to Arduino on {self.port}")
        except Exception as e:
            print(f"Failed to connect to Arduino: {e}")

    def send_control(self, steering, throttle, brake):
        if self.conn and self.conn.is_open:
            cmd = f"S:{steering},T:{throttle},B:{brake}\n"
            self.conn.write(cmd.encode('utf-8'))

    def read_sensors(self):
        if self.conn and self.conn.is_open and self.conn.in_waiting > 0:
            try:
                return self.conn.readline().decode('utf-8').strip()
            except:
                return None
        return None

    def get_sensor_data(self):
        """Parse raw serial string into structured dict."""
        raw = self.read_sensors()
        data = {'left': '--', 'right': '--', 'obd_speed': 0}
        if not raw:
            return data
            
        try:
            # Example: "L:120,R:150,V:45"
            parts = raw.split(',')
            for p in parts:
                if ':' in p:
                    k, v = p.split(':')
                    if k == 'L': data['left'] = float(v)
                    elif k == 'R': data['right'] = float(v)
                    elif k == 'V': data['obd_speed'] = float(v)
        except:
            pass
        return data
