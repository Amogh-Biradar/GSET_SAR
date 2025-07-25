import smbus2
import time
import math, socket

# MPU6050 Registers
cord_x = 0.0
cord_y = 0.0
HEADING = 50.0
METERS = 50.0
velocity_x = 0.0
velocity_y = 0.0
ACCEL_SCALE = 16384.0  # For ±2g range
G = 9.81               # Acceleration due to gravity (m/s²)
REST_ACCEL_X = -868
REST_ACCEL_Y = -88
dt = 1.0  # Time step in seconds

MPU6050_ADDR = 0x68
PWR_MGMT_1 = 0x6B
ACCEL_XOUT_H = 0x3B
GYRO_XOUT_H = 0x43
gyro_available = False

bus = smbus2.SMBus(1)  # 1 for Pi 0/2/3/4
server_address = ('172.20.10.2', 12345)  # Replace with PC's IP and port
client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Wake up MPU6050
bus.write_byte_data(MPU6050_ADDR, PWR_MGMT_1, 0)

def read_word(reg):
    high = bus.read_byte_data(MPU6050_ADDR, reg)
    low = bus.read_byte_data(MPU6050_ADDR, reg+1)
    val = (high << 8) + low
    if val >= 0x8000:
        val = -((65535 - val) + 1)
    return val

def get_axes():
    global velocity_x, velocity_y
    
    # Accelerometer
    raw_accel_x = read_word(ACCEL_XOUT_H)
    accel_x_ms2 = ((raw_accel_x - REST_ACCEL_X) / ACCEL_SCALE) * G

    raw_accel_y = read_word(ACCEL_XOUT_H+2)
    accel_y_ms2 = ((raw_accel_y - REST_ACCEL_Y) / ACCEL_SCALE) * G
    # Gyroscope
    gyro_x = read_word(GYRO_XOUT_H)
    gyro_y = read_word(GYRO_XOUT_H+2)

    # Only integrate velocity if acceleration is above threshold to prevent drift
    movement_threshold = 0.2  # 0.2 m/s² threshold
    if abs(accel_x_ms2) > movement_threshold:
        velocity_x += accel_x_ms2 * dt
    if abs(accel_y_ms2) > movement_threshold:
        velocity_y += accel_y_ms2 * dt
    
    # Apply velocity damping to prevent drift accumulation
    velocity_x *= 0.95  # 5% velocity decay each update
    velocity_y *= 0.95
    
    # Reset tiny velocities to zero
    if abs(velocity_x) < 0.001:
        velocity_x = 0.0
    if abs(velocity_y) < 0.001:
        velocity_y = 0.0

    return {
        'accel_x': accel_x_ms2,
        'accel_y': accel_y_ms2,
        'velo_x': velocity_x,
        'velo_y': velocity_y,
    }

def calculate_navigation():
    global HEADING, METERS, cord_x, cord_y
    
    # Calculate target position using degrees directly
    target_x = METERS * math.cos(math.radians(HEADING))
    target_y = METERS * math.sin(math.radians(HEADING))
    
    # Calculate distance to target
    dx = target_x - cord_x
    dy = target_y - cord_y
    distance_to_target = math.sqrt(dx**2 + dy**2)
    
    # Calculate heading to target (in degrees)
    if distance_to_target > 0.01:  # Only calculate if we're not essentially at target
        heading_to_target = math.degrees(math.atan2(dy, dx))
        # Normalize heading to 0-360 degrees
        if heading_to_target < 0:
            heading_to_target += 360
    else:
        heading_to_target = 0
        distance_to_target = 0
    
    # Calculate straight-line distance if traveling at the required heading
    straight_line_distance = distance_to_target
    
    return distance_to_target, heading_to_target, straight_line_distance

def alertBase(message):
    """
    Send messages to the base station in the expected format.
    The Base_Station.py expects specific message formats:
    - Position updates: "Position: X=0.00m, Y=0.00m"
    - Target info: "Target: 0.00m at 0.0° | Distance remaining: 0.00m | Heading needed: 0.0°"
    - Straight-line distance: "Straight-line distance to target: 0.00m"
    - Alerts: Messages with "IMPORTANT", "scream detected", or "person found"
    """
    # Make sure the message is properly formatted
    if "Position:" not in message and "X=" in message and "Y=" in message:
        message = f"Position: {message}"
    
    # Ensure we're sending to the correct address
    try:
        client_socket.sendto(message.encode(), server_address)
        # Send twice to reduce chance of packet loss
        client_socket.sendto(message.encode(), server_address)
        print(f"✓ Sent to base station: {message}")
    except Exception as e:
        print(f"✗ Error sending message to base station: {e}")
        print(f"  Message was: {message}")
        print(f"  Server address: {server_address}")

def update_position():
    global cord_x, cord_y, velocity_x, velocity_y
    
    axes = get_axes()
    
    # Convert raw accelerometer values to m/s² (approximate conversion)
    # MPU6050 default range is ±2g, so divide by ~16384 to get g-force, then multiply by 9.81
    accel_x_ms2 = axes['accel_x']
    accel_y_ms2 = axes['accel_y']
    velocity_x = axes['velo_x']
    velocity_y = axes['velo_y']
    
    # Update position: x = x0 + v*t + 0.5*a*t²
    cord_x += velocity_x * dt + 0.5 * accel_x_ms2 * dt * dt
    cord_y += velocity_y * dt + 0.5 * accel_y_ms2 * dt * dt

    # Reset position if it gets unreasonably large (beyond 100 meters)
    if abs(cord_x) > 100.0 or abs(cord_y) > 100.0:
        print(f"WARNING: Position reset due to excessive drift. Was at ({cord_x:.2f}, {cord_y:.2f})m")
        cord_x = 0.0
        cord_y = 0.0
        velocity_x = 0.0
        velocity_y = 0.0
    
    # Calculate navigation to target
    distance_remaining, heading_needed, straight_line_distance = calculate_navigation()
    
    print(f"Position: X={cord_x:.2f}m, Y={cord_y:.2f}m | Velocity: ({velocity_x:.3f}, {velocity_y:.3f})m/s | Accel: ({accel_x_ms2:.3f}, {accel_y_ms2:.3f})m/s²")
    print(f"Target: {METERS}m at {HEADING}° | Distance remaining: {distance_remaining:.2f}m | Heading needed: {heading_needed:.1f}°")
    print(f"Straight-line distance to target: {straight_line_distance:.2f}m")
    print("-" * 80)

    position_msg = f"Position: X={cord_x:.2f}m, Y={cord_y:.2f}m"
    print(f"Sending: {position_msg}")
    alertBase(position_msg)
    
    # Send navigation data in the format expected by Base_Station.py
    status = "Real sensor" if gyro_available else "Simulated"
    message1 = f"Target: {METERS}m at {HEADING}° | Distance remaining: {distance_remaining:.2f}m | Heading needed: {heading_needed:.1f}° ({status})"
    message2 = f"Straight-line distance to target: {straight_line_distance:.2f}m"
    print(f"Sending: {message1}")
    print(f"Sending: {message2}")
    alertBase(message1)
    alertBase(message2)

if __name__ == "__main__":
    try:
        while True:
            update_position()
            # Get updated navigation after each position update
            distance_remaining, heading_needed, straight_line_distance = calculate_navigation()
            
            # Break if we're close enough to target (within 1cm)
            if straight_line_distance <= 0.01:
                print(f"Target reached! Final position: X={cord_x:.2f}m, Y={cord_y:.2f}m")
                break
                
            time.sleep(1.0)  # Update every second
    except KeyboardInterrupt:
        print("Exiting...")
        print(f"Final position: X={cord_x:.2f}m, Y={cord_y:.2f}m")

