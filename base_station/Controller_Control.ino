void setup() {
  Serial.begin(9600);
}

void loop() {
  if (Serial.available()) {
    String message = Serial.readStringUntil('\n');  // Read line
    parseAndExecute(message);
  }
}

void parseAndExecute(String msg) {
  // Split by commas
  int lastIndex = 0;
  while (lastIndex < msg.length()) {
    int nextIndex = msg.indexOf(',', lastIndex);
    if (nextIndex == -1) nextIndex = msg.length();

    String command = msg.substring(lastIndex, nextIndex);
    int colonIndex = command.indexOf(':');
    if (colonIndex != -1) {
      int pin = command.substring(0, colonIndex).toInt();
      int value = command.substring(colonIndex + 1).toInt();
      analogWrite(pin, value);
    }

    lastIndex = nextIndex + 1;
  }
}

