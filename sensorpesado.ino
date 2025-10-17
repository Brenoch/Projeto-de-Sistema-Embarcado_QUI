#define TRIG 9
#define ECHO 10
#define BUZZER 8
#define LED 7

void setup() {
  pinMode(TRIG, OUTPUT);
  pinMode(ECHO, INPUT);
  pinMode(BUZZER, OUTPUT);
  pinMode(LED, OUTPUT);
  Serial.begin(9600);
}

long medir_distancia() {
  digitalWrite(TRIG, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG, LOW);
  long duration = pulseIn(ECHO, HIGH, 20000);
  long distance_cm = duration / 58;
  return distance_cm;
}

void loop() {
  if (Serial.available()) {
    char comando = Serial.read();
    if (comando == '1') {
      digitalWrite(BUZZER, HIGH);
      digitalWrite(LED, HIGH);
    } else if (comando == '0') {
      digitalWrite(BUZZER, LOW);
      digitalWrite(LED, LOW);
    }
  }

  long distancia = medir_distancia();
  if (distancia > 0 && distancia <= 100) {
    digitalWrite(BUZZER, HIGH);
    digitalWrite(LED, HIGH);
  } else {
    digitalWrite(BUZZER, LOW);
    digitalWrite(LED, LOW);
  }
  
  delay(200);
}
