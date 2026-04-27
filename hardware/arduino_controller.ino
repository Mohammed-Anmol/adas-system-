// Arduino controller for real-time PWM and sensor reading
// Handles Steering, Throttle, Braking, and Ultrasonic sensors

const int STEERING_PIN = 9;
const int THROTTLE_PIN = 10;
const int BRAKE_PIN = 11;
const int ULTRASONIC_TRIG_PIN = 4;
const int ULTRASONIC_ECHO_PIN = 5;

void setup() {
  Serial.begin(115200);
  pinMode(STEERING_PIN, OUTPUT);
  pinMode(THROTTLE_PIN, OUTPUT);
  pinMode(BRAKE_PIN, OUTPUT);
  
  pinMode(ULTRASONIC_TRIG_PIN, OUTPUT);
  pinMode(ULTRASONIC_ECHO_PIN, INPUT);
}

void loop() {
  // Read ultrasonic for blind spot
  long duration, distance;
  digitalWrite(ULTRASONIC_TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(ULTRASONIC_TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(ULTRASONIC_TRIG_PIN, LOW);
  duration = pulseIn(ULTRASONIC_ECHO_PIN, HIGH);
  distance = (duration / 2) / 29.1;

  // Send sensor data to Pi
  Serial.print("D:");
  Serial.println(distance);

  // Read commands from Pi
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    processCommand(cmd);
  }
  
  delay(10);
}

void processCommand(String cmd) {
  // Format expected: "S:90,T:120,B:0"
  // Stub function to parse and apply PWM
}
