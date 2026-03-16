// =============================
// Bluetooth RC Car
// Commands:
// 0 = stop
// 1 = drive forward straight
// 2 = drive forward right curve
// 3 = drive forward left curve
// 4 = drive backward straight
// 5 = keep previous command
// =============================

// PWM speed pins
// decide "how much motors are driven"
#define Lpwm_pin 6   // left side
#define Rpwm_pin 5   // right side

// Motor direction pins
// F = forward, B = backward
// decide what direction the motors spins
#define pinLB 2   // left side backward 
#define pinLF 4   // left side forward
#define pinRB 7   // right side backward
#define pinRF 8   // right side forward

char bluetooth_data = '0';

// -----------------------------
// Setup motor pins
// -----------------------------
// all pins output
void M_Control_IO_config()
{
  pinMode(pinLB, OUTPUT);
  pinMode(pinLF, OUTPUT);
  pinMode(pinRB, OUTPUT);
  pinMode(pinRF, OUTPUT);
  pinMode(Lpwm_pin, OUTPUT);
  pinMode(Rpwm_pin, OUTPUT);
}

// -----------------------------
// Set left and right speed separately
// range: 0..255
// -----------------------------
void Set_Speed(int leftSpeed, int rightSpeed)
{
  analogWrite(Lpwm_pin, leftSpeed);
  analogWrite(Rpwm_pin, rightSpeed);
}

// -----------------------------
// Movements
// -----------------------------
void stopCar()
{
  digitalWrite(pinLB, HIGH);
  digitalWrite(pinLF, HIGH);
  digitalWrite(pinRB, HIGH);
  digitalWrite(pinRF, HIGH);
  Set_Speed(0, 0);
}

void forwardStraight(int speedVal)
{
  digitalWrite(pinLB, LOW);
  digitalWrite(pinLF, HIGH);
  digitalWrite(pinRB, LOW);
  digitalWrite(pinRF, HIGH);
  Set_Speed(speedVal, speedVal);
}

void backwardStraight(int speedVal)
{
  digitalWrite(pinLB, HIGH);
  digitalWrite(pinLF, LOW);
  digitalWrite(pinRB, HIGH);
  digitalWrite(pinRF, LOW);
  Set_Speed(speedVal, speedVal);
}

void Curve(int left_speed, int right_speed)
{
  // Left side faster, right side slower
  digitalWrite(pinLB, LOW);
  digitalWrite(pinLF, HIGH);
  digitalWrite(pinRB, LOW);
  digitalWrite(pinRF, HIGH);
  Set_Speed(left_speed, right_speed);
}

// -----------------------------
// Arduino setup
// -----------------------------
void setup()
{
  M_Control_IO_config();
  Serial.begin(9600);
  stopCar();
}

// -----------------------------
// Main loop
// -----------------------------
void loop()
{

  // read bluetooth
  if (Serial.available())
  {
    bluetooth_data = Serial.read();
  }

  switch (bluetooth_data)
  {
    // STOP
    case '0':
      stopCar();
      break;

    // FORWARD
    case '1':
      forwardStraight(220);
      break;

    // FORWARD RIGHT
    case '2':
      Curve(220, 110);
      break;

    // FORWARD LEFT
    case '3':
      Curve(110, 220);
      break;

    // BACKWARDS
    case '4':
      backwardStraight(200);
      break;

    // OTHER -> stop
    case '5':
      // TODO: think of OTHER logic
      // currently OTHER = STOP
      stopCar();
      break;

    default:
      break;
  }
}