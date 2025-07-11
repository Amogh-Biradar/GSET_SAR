#define p1Pin A0 //fly up and down
#define p2Pin A1 //spin left and right
#define p3Pin A2 //move forward and backward
#define p4Pin A3 //move left and right

int timeToSpinOneDegree = 0; //NEED TO CALIBRATE
int timeToMoveOneCMHor = 0; ////NEED TO CALIBRATE
int timeToMoveOneCMVer = 0; ////NEED TO CALIBRATE


void setup() {
  Serial.begin(115200);
  pinMode(A0, OUTPUT);
  pinMode(A1, OUTPUT);
  pinMode(A2, OUTPUT);
  pinMode(A3, OUTPUT);


  while(!Serial) {

  }

}

void loop() {
  char data = 'D';

  data = readCharFromSerial();

  switch(data) {
    // case '1':
    //   flightPath1();
    //   break;
    case '2':
      flightPath2();
      flyInPlace();
      break;
    case 'S':
      spinToAngle();
      flyInPlace();
      break;
    case 'I':
      spinInfinite();
      flyInPlace();
      break;
    case 'M':
      moveInfinite();
      flyInPlace();
      break;
    case 'D':
    default:
      flyInPlace();
      break;
    
  }



}

//move down and forward at the same time
void flightPath1() {
  // flyInPlace();
  // int val1 = readNumberFromSerial();
  // int val2 = 512;
  // int val3 = readNumberFromSerial();
  // int val4 = 512;
  // // int delayMS = timeToMove
  // analogWrite(p1Pin, val1);
  // analogWrite(p2Pin, val2);
  // analogWrite(p3Pin, val3);
  // analogWrite(p4Pin, val4);
  // delay(delayMS);

}

//move forward and then once above target, move down
void flightPath2() {

  //Move Forward
  flyInPlace();
  int val1 = 512;
  int val2 = 512;
  int val3 = 600;
  int val4 = 512;
  int delayMS = readNumberFromSerial() * timeToMoveOneCMHor;
  analogWrite(p1Pin, val1);
  analogWrite(p2Pin, val2);
  analogWrite(p3Pin, val3);
  analogWrite(p4Pin, val4);
  delay(delayMS);

  //Move Down
  flyInPlace();
  val1 = 450;
  val2 = 512;
  val3 = 512;
  val4 = 512;
  delayMS = readNumberFromSerial() * timeToMoveOneCMVer;
  analogWrite(p1Pin, val1);
  analogWrite(p2Pin, val2);
  analogWrite(p3Pin, val3);
  analogWrite(p4Pin, val4);
  delay(delayMS);
}

void flyInPlace() {
  analogWrite(p1Pin, 512);
  analogWrite(p2Pin, 512);
  analogWrite(p3Pin, 512);
  analogWrite(p4Pin, 512);
} 


int readNumberFromSerial() {
  while (Serial.available() == 0) {
    // Wait until data is available
  }
  Serial.println("Recieved");
  int x = Serial.parseInt();
  return x;  // Waits until an integer is fully read
}

char readCharFromSerial() {
  while (Serial.available() == 0) {
    // Wait for input
  }
  Serial.println("Recieved");
  char x = Serial.read();
  return x;  // Read one byte (character)

}


void spinToAngle() {
  int ms = readNumberFromSerial() * timeToSpinOneDegree;
  analogWrite(p1Pin, 512);
  analogWrite(p2Pin, 600);
  analogWrite(p3Pin, 512);
  analogWrite(p4Pin, 512);
  delay(ms);
  flyInPlace();
}

void spinInfinite() {
  analogWrite(p1Pin, 512);
  analogWrite(p2Pin, 600);
  analogWrite(p3Pin, 512);
  analogWrite(p4Pin, 512);

  while (Serial.available() <= 0) {
    
  }
  char flag = readCharFromSerial(); 
  return;
}

void moveInfinite() {
  analogWrite(p1Pin, 512);
  analogWrite(p2Pin, 512);
  analogWrite(p3Pin, 600);
  analogWrite(p4Pin, 512);

  while (Serial.available() <= 0) {
    
  }
  int flag = readCharFromSerial(); 
  return;
  
}