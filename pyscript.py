'''
TO DO:
1. localization + get coords/instructions
2. gyroscope, sensors, update coords/instructions
3. base station sending and interfacing with user
4. drone assembly, desoldering, weight testing
5. paper
'''

from gcc_phat import gcc_phat
from scipy.optimize import minimize_scalar
import sounddevice as sd
import numpy as np
import Adafruit_GPIO. spi as SPI
import Adafruit_MCP3008
import torch
import torch.nn as nn
import time
import torch.nn.functional as F
import time
from RF_old_np import AzimuthRandomForest
import soundfile as sf
import noisereduce as nr
from audioSim import Environment, Wave 
import math
import smbus2
import socket
import spidev
from scipy.io.wavfile import write

# ====== CONFIG ======
DEVICE_IDS = [1, 2, 3]  # hardware IDs for USB mics (switch if necessary)
CLK = 23
MISO = 21
MOSI = 19
CS = 26
mcp = Adafruit_MCP3008.MCP3008(clk = CLK, cs = CS, miso = MISO, mosi = MOSI)
ANALOGPORT = [0, 1, 2]
SAMPLE_RATE = 10000
RECORD_SECONDS = 9.072
FILTER_PATH = 'denoiser_model.pth'
SCREAM_PATH = 'scream_classifier.pth'
HEADING = -1
LAST_HEADING = None
METERS = -1
server_address = ('172.20.10.2', 12345)  # Replace with PC's IP and port
client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

#GYRO CONFIG
MPU6050_ADDR = 0x68
PWR_MGMT_1 = 0x6B
ACCEL_XOUT_H = 0x3B
GYRO_XOUT_H = 0x43

bus = smbus2.SMBus(1)  # 1 for Pi 0/2/3/4

# Wake up MPU6050
bus.write_byte_data(MPU6050_ADDR, PWR_MGMT_1, 0)

# ====== FILTER MODEL ======
class Denoiser1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(1, 16, 15, stride=1, padding=7),
            nn.ReLU(),
            nn.Conv1d(16, 64, 15, stride=2, padding=7),
            nn.ReLU(),
            nn.Conv1d(64, 256, 15, stride=2, padding=7),
            nn.ReLU(),
            nn.Conv1d(256, 512, 15, stride=2, padding=7),
            nn.ReLU(),
            nn.ConvTranspose1d(512, 256, 15, stride=2, padding=7, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(256, 64, 15, stride=2, padding=7, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 16, 15, stride=2, padding=7, output_padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 1, 15, stride=1, padding=7),
            nn.Tanh()  # constrain output to [-1, 1]
        )

    def forward(self, x):
        return self.model(x)
    
# ====== SCREAM DETECTION MODEL ======
class ScreamClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(16, 64, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(64, 128, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(128, 256, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.AdaptiveAvgPool1d(128),  # 128
            nn.Flatten(),
            nn.Linear(256 * 128, 64), # 256, 128, 64
            nn.ReLU(),
            nn.Linear(64, 1), # 64
            nn.Sigmoid()
        )

    def forward(self, x):
        for layer in self.net:
            x = layer(x)
            print(f"{layer.__class__.__name__}: {x.shape}")
        return x
    
# ====== AUDIO CAPTURE ======
# Configuration
SAMPLE_RATE = 10000  # samples per second per channel
DURATION = 5         # seconds to record
VREF = 3.3           # MCP3008 reference voltage
BITS = 10            # MCP3008 resolution

# Initialize SPI
spi = spidev.SpiDev()
spi.open(0, 0)  # Bus 0, Device (CS) 0
spi.max_speed_hz = 1000000  # 1 MHz SPI clock

model = AzimuthRandomForest()
model.load_model('azimuth_np_old.pkl')

def read_adc(channel):
    # MCP3008 SPI protocol: 3 bytes transaction
    # Start bit, single-ended/differential bit, channel selection bits
    # Format: 00000001 | (S/D + channel bits << 4) | 00000000
    if channel < 0 or channel > 7:
        return -1
    start_bit = 0x01
    sgl_diff = 0x08  # single-ended
    command = start_bit << 16 | (sgl_diff | channel) << 12
    # Build the 3 bytes to send
    tx = [1, (8 + channel) << 4, 0]
    rx = spi.xfer2(tx)
    # Extract 10-bit result from returned bytes
    result = ((rx[1] & 3) << 8) | rx[2]
    return result

def adc_to_voltage(adc_value):
    return (adc_value * VREF) / (2**BITS - 1)

def main_audio():

    num_samples = int(SAMPLE_RATE * DURATION)
    # Preallocate arrays
    data = {ch: np.zeros(num_samples, dtype=np.float32) for ch in ANALOGPORT}

    print(f"Recording {DURATION} seconds at {SAMPLE_RATE} Hz per channel...")
    start_time = time.time()

    for i in range(num_samples):
        for ch in ANALOGPORT:
            # Read raw ADC (0–1023)
            raw = mcp.read_adc(ch)
            data[ch][i] = raw
            #add a delay        
            # pace sampling
        elapsed = time.time() - start_time
        target = (i + 1) / SAMPLE_RATE
        to_sleep = target - elapsed
        if to_sleep > 0:
            time.sleep(to_sleep)

    print("Recording complete.")

    filenames = []
    # process & save each channel
    for ch in ANALOGPORT:
        # Convert 0–1023 → -1.0–1.0
        norm = 2 * (data[ch] / (2**BITS - 1) - 0.5)
        # Software gain
        gain = 5.0
        amp = np.clip(norm * gain, -1.0, 1.0)

        fname = f"mic_channel_{ch}.wav"
        sf.write(fname, amp, SAMPLE_RATE)
        print(f"Saved {fname}")
        filenames.append(fname)

    return tuple(filenames)
   # return filenames


# ====== AUDIO PROCESSING ======
def preprocess_wave(waveform):
    waveform = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if waveform.shape[1] > 400000:
        waveform = waveform[:, :400000]
    else:
        waveform = F.pad(waveform, (0, 400000 - waveform.shape[1]))
    return waveform

# ====== LOAD MODELS ======
filterModel = Denoiser1D()
filterModel.load_state_dict(torch.load(FILTER_PATH, map_location='cpu'))
filterModel.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
screamModel = ScreamClassifier().to(device)
screamModel.load_state_dict(torch.load(SCREAM_PATH, map_location=device))
screamModel.eval()

# ====== AUDIO FUNCTION ======
def record_and_filter(filename):
    print(f"[INFO] processing from file {filename}...")
    audio = preprocess_wave(torch.tensor(filename))

    # === Denoise using 1D CNN ===
    with torch.no_grad():
        output_wave = filterModel(audio.unsqueeze(0)).squeeze(0)
        output_np = output_wave.squeeze().detach().numpy()
        output_denoised = nr.reduce_noise(y=output_np, sr=SAMPLE_RATE, prop_decrease=0.75)
        output_wave = torch.tensor(output_denoised, dtype=torch.float32)
        output_wave = output_wave / torch.max(torch.abs(output_wave))  # normalize to [-1, 1]
        output_wave = output_wave.unsqueeze(0)
    return output_wave

# ====== Scream Detection ======
def class_scream(wavIdx):
    alert = False
    res = [False, False, False]
    conf = [0, 0, 0]

    for i in range(3):
        if wavIdx[i].dim() == 2:
            wavIdx[i] = wavIdx[i].unsqueeze(0)

        with torch.no_grad():
            pred_raw = screamModel(wavIdx[i]).item()

            if pred_raw > 0.5 and alert == False:
                alert = True
                # Send position first so the UI can update the location
                position_msg = f"Position: {getCoords()}"
                alertBase(position_msg)
                # Then send the alert in the format expected by Base_Station.py
                alertBase(f"IMPORTANT: Human scream detected ({pred_raw * 100:.2f}% confidence). Drone at coordinates: {getCoords()}.")
            if pred_raw > 0.5:
                res[i] = True
            conf[i] = pred_raw
    
    # If any microphone detected a scream, localize it
    if res[0] or res[1] or res[2]:
        localize(wavIdx)

# ====== Localization if Scream Detected ======
def localize(wavIdx):
    global HEADING, METERS
    mic_positions = [[0, 0], [0.05, 0], [0.025, 0.0433]]

    fs_a, sig_a = wavIdx[0]
    fs_b, sig_b = wavIdx[1]
    fs_c, sig_c = wavIdx[2]

    tdoa_ab = gcc_phat(sig_a, sig_b, fs_a)[0]
    tdoa_ac = gcc_phat(sig_a, sig_c, fs_a)[0]
    
    
    HEADING = model.predict(tdoa_ab * 1e6, tdoa_ac * 1e6)
    if abs(HEADING - LAST_HEADING) > 150:
        position_msg = f"Position: {getCoords()}"
        alertBase(position_msg)
        
    #METERS = getEstDist([tdoa_ab, tdoa_ac], HEADING, mic_positions)

    # Send alert in format expected by Base_Station.py
    
    alertBase(f"IMPORTANT: Localized scream at heading {HEADING:.1f}° and distance {METERS:.2f}m.")
    
    # Navigation loop
    while METERS > 3:
        gyro()  # This will send position and navigation updates
        time.sleep(1)
    
    # Send person found alert in format expected by Base_Station.py
    position_msg = f"Position: {getCoords()}"
    alertBase(position_msg)
    alertBase(f"IMPORTANT: Person found at coordinates: {getCoords()}.")


# # ====== Image Capture ======
# def takeImg():
#     cap = cv2.VideoCapture(0)  # 0 is the default camera index
#     if not cap.isOpened():
#         print("Error: Could not open camera.")
#         return None

#     ret, frame = cap.read()
#     cap.release()

#     if not ret:
#         print("Error: Failed to capture image.")
#         return None

#     cv2.imwrite('capture.jpg', frame)
#     return frame

# # ====== Image Classification ======
# def classImage(image):
#     return (f"Here class the image with mobilenet and potentially other network. Found at coords: {getCoords()}.")

# ====== Stabilize with Gyroscope ======

def read_word(reg):
    high = bus.read_byte_data(MPU6050_ADDR, reg)
    low = bus.read_byte_data(MPU6050_ADDR, reg+1)
    val = (high << 8) + low
    if val >= 0x8000:
        val = -((65535 - val) + 1)
    return val

def get_axes():
    # Accelerometer
    accel_x = read_word(ACCEL_XOUT_H)
    accel_y = read_word(ACCEL_XOUT_H+2)
    accel_z = read_word(ACCEL_XOUT_H+4)
    # Gyroscope
    gyro_x = read_word(GYRO_XOUT_H)
    gyro_y = read_word(GYRO_XOUT_H+2)
    gyro_z = read_word(GYRO_XOUT_H+4)
    return {
        'accel_x': accel_x,
        'accel_y': accel_y,
        'accel_z': accel_z,
        'gyro_x': gyro_x,
        'gyro_y': gyro_y,
        'gyro_z': gyro_z
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

def gyro():
    global cord_x, cord_y, velocity_x, velocity_y, distance_remaining, heading_needed, straight_line_distance
    
    # Initialize global variables if they don't exist
    if 'cord_x' not in globals():
        global cord_x, cord_y, velocity_x, velocity_y
        cord_x = 0.0
        cord_y = 0.0                    #CHECK THIS
        velocity_x = 0.0
        velocity_y = 0.0
    
    axes = get_axes()
    
    # Convert raw accelerometer values to m/s² (approximate conversion)
    # MPU6050 default range is ±2g, so divide by ~16384 to get g-force, then multiply by 9.81
    accel_x_ms2 = (axes['accel_x'] / 16384.0) * 9.81
    accel_y_ms2 = (axes['accel_y'] / 16384.0) * 9.81
    
    # Time step (1 second)
    dt = 1.0
    
    # Update velocity: v = v0 + a*t
    velocity_x += accel_x_ms2 * dt
    velocity_y += accel_y_ms2 * dt
    
    # Update position: x = x0 + v*t + 0.5*a*t²
    cord_x += velocity_x * dt + 0.5 * accel_x_ms2 * dt * dt
    cord_y += velocity_y * dt + 0.5 * accel_y_ms2 * dt * dt
    
    # Calculate navigation to target
    distance_remaining, heading_needed, straight_line_distance = calculate_navigation()
    
    # Send position update first - this is what Base_Station.py expects
    position_msg = f"Position: X={cord_x:.2f}m, Y={cord_y:.2f}m"
    alertBase(position_msg)
    
    # Send navigation data in the format expected by Base_Station.py
    message1 = f"Target: {METERS}m at {HEADING}° | Distance remaining: {distance_remaining:.2f}m | Heading needed: {heading_needed:.1f}°"
    message2 = f"Straight-line distance to target: {straight_line_distance:.2f}m"
    alertBase(message1)
    alertBase(message2)

# ====== Alert Base Station ======
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
    except Exception as e:
        print(f"Error sending message: {e}")

# # ====== Image Sending ======
# def sendImg(image):
#     print(image)

# # ====== Audio Sending ======
# def sendAudio(audio):
#     return

# ====== Get Coordinates ======
def getCoords():
    """Return coordinates in the format expected by Base_Station.py"""
    return f"X={cord_x:.2f}m, Y={cord_y:.2f}m"
 

# # ====== Receive from Base Station ======

#last_trigger_time = time.time()
#interval = 31  # seconds
# ====== MAIN LOOP ======

# Initialize global variables
cord_x = 0.0
cord_y = 0.0
velocity_x = 0.0
velocity_y = 0.0
distance_remaining = 0.0
heading_needed = 0.0
straight_line_distance = 0.0

# Initialize audio processing variables
wavIdx = []
filtered_wavIdx = []

# Initialize gyroscope
try:
    read_word(ACCEL_XOUT_H)  # Test read to ensure gyro is working
    get_axes()  # Initialize gyro readings
    gyro_status = "Gyroscope initialized successfully."
except Exception as e:
    gyro_status = f"Gyroscope initialization error: {str(e)}"
    print(gyro_status)

# Send initial position to Base Station
initial_position = f"Position: X={cord_x:.2f}m, Y={cord_y:.2f}m"
alertBase(initial_position)
alertBase(f"System initialized and ready. {gyro_status}")

# Main loop
try:
    while True:
        try:
            # Gyroscope for flight stabilization and position updates
            gyro()  # This will send position updates to Base_Station.py
            
            # Audio processing
            filename_ch0, filename_ch1, filename_ch2, *_ = main_audio()
            wavIdx = [filename_ch0, filename_ch1, filename_ch2]
            filtered_wavIdx = []  # Clear previous data
            for i in range(3):
                data, _ = sf.read(wavIdx[i])  # Load waveform data
                filtered = preprocess_wave(data)       # Now pass actual data
                filtered_wavIdx.append(filtered)
            class_scream(filtered_wavIdx)
            
            # Brief pause to prevent overwhelming the network
            time.sleep(0.1)
        except Exception as e:
            error_msg = f"Error in main loop: {str(e)}"
            print(error_msg)
            alertBase(f"IMPORTANT: {error_msg}")
finally:
    spi.close()
    

    # # Image Processes
    # current_time = time.time()
    # if current_time - last_trigger_time >= interval:
    #     image = takeImg()
    #     message = classImage(image)
    #     alertBase(message)
    #     if "person" in message:
    #         alertBase(f"IMPORTANT: PERSON SPOTTED AT COORDS: {getCoords()}.")
    #         sendImg(image)
    #     if "signal flare" in message:
    #         alertBase(f"IMPORTANT: FLARE SPOTTED AT COORDS: {getCoords()}.")
    #         sendImg(image)
    #     last_trigger_time = current_time
    # # Optional: sleep briefly to reduce CPU usage
    # time.sleep(0.1)
    
    # Audio Capture
    
    
    


    # try:
    #     for idk, dev_id in enumerate(DEVICE_IDS):
    #         wavIdx.append(record_and_filter(dev_id)) # [1, N]

    #         class_scream(wavIdx)

    # except KeyboardInterrupt:
    #     print("Exiting...")
    #     break

    # Special Commands
    # # Take and analyze image
    # if receiveBase() == "1":
    #     alertBase(classImage(takeImg()))
    # # Send image
    # if receiveBase() == "2":
    #     sendImg(takeImg())
    # # Send audio sample
    # if receiveBase() == "3":
    #     sendAudio(wavIdx)
    # # Send coordinates
    # if receiveBase() == "4":
    #     alertBase(f"Coordinates: {getCoords()}")
