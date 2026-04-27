"""
Emergency SMS alert system using Twilio (or serial GSM fallback).
Triggered on detected collision or critical safety events.
"""
import time
import json


class EmergencyAlerter:
    """Sends emergency SMS alerts via Twilio API or GSM module."""

    def __init__(self, config):
        safety = config.get('safety', {})
        self.enabled = safety.get('emergency_sms_enabled', False)
        self.to_number = safety.get('emergency_sms_number', '')
        self.account_sid = safety.get('twilio_account_sid', '')
        self.auth_token = safety.get('twilio_auth_token', '')
        self.from_number = safety.get('twilio_from_number', '')
        self._client = None
        self._last_alert_time = 0
        self._cooldown = 30  # seconds between alerts

        if self.enabled and self.account_sid and self.auth_token:
            try:
                from twilio.rest import Client
                self._client = Client(self.account_sid, self.auth_token)
                print("[Alert] Twilio SMS client initialized.")
            except ImportError:
                print("[Alert] twilio package not installed — SMS disabled.")
        else:
            print("[Alert] Emergency SMS disabled or not configured.")

    def trigger_alert(self, event_type, location=None, extra_data=None):
        """Send an emergency alert.

        Parameters
        ----------
        event_type : str — "collision", "drowsiness", "sensor_fault"
        location   : dict {lat, lon} or None
        extra_data : dict — any extra telemetry to include

        Returns
        -------
        bool : True if alert was sent
        """
        now = time.time()
        if now - self._last_alert_time < self._cooldown:
            return False  # rate-limited

        loc_str = "unknown"
        if location:
            loc_str = f"{location.get('lat', '?')},{location.get('lon', '?')}"

        body = (f"[ADAS EMERGENCY] {event_type.upper()} detected!\n"
                f"Location: {loc_str}\n"
                f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        if extra_data:
            body += f"\nData: {json.dumps(extra_data, default=str)}"

        print(f"[Alert] {body}")

        if self._client and self.to_number:
            try:
                msg = self._client.messages.create(
                    body=body,
                    from_=self.from_number,
                    to=self.to_number,
                )
                print(f"[Alert] SMS sent: {msg.sid}")
                self._last_alert_time = now
                return True
            except Exception as e:
                print(f"[Alert] SMS failed: {e}")
                return False

        self._last_alert_time = now
        return False
