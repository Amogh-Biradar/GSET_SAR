"""Get Tyler's Code to get directions"""

import serial
import time 

arduino = serial.Serial(port='COM4', baudrate=115200, timeout=.1) 
def write_read(x): 
    try:
        arduino.write((x + '\n').encode())
        print(f"Sent '{x}' to Arduino. Waiting for acknowledgment...")

        buffer = ''
        while True:
            if arduino.in_waiting:
                byte = arduino.read().decode('utf-8', errors='ignore')
                if byte in ['\n', '\r']:
                    if buffer.strip() == "Recieved":
                        print("Arduino acknowledged: Recieved ✅")
                        break
                    buffer = ''  # Reset buffer
                else:
                    buffer += byte

        arduino.close()

    except serial.SerialException as e:
        print(f"Serial error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
	
#send cm needed to travel
def main(): 
    cmd = ''
    num = (input("Enter a cmd (Options are 2, S, I, M, D): "))
    num = num[0]
    write_read(num) 
        # Wait until Arduino sends a non-empty message back
    match num:
        case 'S': #spin to angle
            heading = int(input("Enter a heading in degrees: "))# Taking input from user
            write_read(heading)
        case 'I': #spin infinitely
            inp = input("Press Enter to stop spinning and continue...")
            write_read(inp)
        case 'M': #move forward infinitely
            inp = input("Press Enter to stop moving and continue...")
            write_read(inp)
        case '2': #flight path 2
            cm = input("Enter cm to travel forward: ")
            write_read(cm)
            cm = input("Enter cm to travel down")
            write_read(cm)
        #case 'D' does not exist because the FlyInPlace() method does not require extra parameters
    main()
main()



