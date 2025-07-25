import time
import math
import numpy as np
import soundfile as sf
import socket

# ===== SIMULATION + REAL-WIFI =====
# Server for Base Station (replace with actual address/port)
server_address = ('172.20.10.2', 12345)
client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Record the simulation start time for gyro calculations
t0 = time.time()

# ===== STUBS =====
# Simulate gyro by sending random coordinates in main code format
import random

def gyro():
    # Simulate drone gyro/navigation: send position and navigation updates
    # Randomize coordinates and navigation metrics
    x = random.uniform(-5.0, 5.0)
    y = random.uniform(-5.0, 5.0)
    distance_remaining = random.uniform(0.0, 10.0)
    heading_needed = random.uniform(0.0, 360.0)
    straight_line_distance = random.uniform(0.0, 10.0)

    # Position update
    position_msg = f"Position: X={x:.2f}m, Y={y:.2f}m"
    client_socket.sendto(position_msg.encode(), server_address)
    print(f"[SIM GYRO] Sent: {position_msg}")

    # Navigation data
    message1 = f"Target: {distance_remaining:.2f}m at {heading_needed:.1f}° | Distance remaining: {distance_remaining:.2f}m | Heading needed: {heading_needed:.1f}°"
    message2 = f"Straight-line distance to target: {straight_line_distance:.2f}m"
    client_socket.sendto(message1.encode(), server_address)
    client_socket.sendto(message2.encode(), server_address)
    print(f"[SIM GYRO] Sent: {message1}")
    print(f"[SIM GYRO] Sent: {message2}")

# Stub for alertBase – forward messages over UDP as in main script
def alertBase(msg):
    client_socket.sendto(msg.encode(), server_address)
    print(f"[SIM ALERT] Sent: {msg}")

# Local stubs for audio/ML parts

def preprocess_wave(data):
    # Simulated preprocessing (no-op)
    print("[SIM] Preprocessing waveform")
    return data

# Simulated denoising stub
import numpy as np

def record_and_filter(filename):
    print(f"[SIM] Denoising {filename}")
    # Load the file to determine length
    data, sr = sf.read(filename)
    length = len(data)
    # Return a random 'tensor-like' array
    return np.random.rand(1, length)

# Simulated scream detection stub
def class_scream(waves):
    detected = False
    for i, wav in enumerate(waves):
        if np.random.rand() > 0.8:
            msg = f"IMPORTANT: Simulated scream detected on mic {i}"
            alertBase(msg)
            detected = True
    if not detected:
        print("SCREAM_NOT_DETECTED")

# record_and_filter and class_scream remain unchanged

# ===== Simulated main_audio =====
ANALOGPORT = [0, 1, 2]
SAMPLE_RATE = 10000
DURATION = 1  # seconds


def main_audio():
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    filenames = []
    for ch in ANALOGPORT:
        freq = 440 + ch * 100  # different tone per channel
        sig = np.sin(2 * math.pi * freq * t).astype(np.float32)
        fname = f"sim_channel_{ch}.wav"
        sf.write(fname, sig, SAMPLE_RATE)
        print(f"[SIM AUDIO] Saved {fname}")
        filenames.append(fname)
    return tuple(filenames)


# ===== MAIN LOOP =====
try:
    last_scream_time = 0
    while True:
        try:
            # Simulate gyro and send real UDP
            gyro()

            # Simulate audio capture
            f0, f1, f2 = main_audio()
            wav_files = [f0, f1, f2]
            processed = []
            for fn in wav_files:
                data, sr = sf.read(fn, dtype='float32')
                processed.append(preprocess_wave(data))

            # Simulated denoise + detection
            denoised = [record_and_filter(fn) for fn in wav_files]
            class_scream(denoised)

            # 10% probability scream alert, at most once every ~15 seconds
            if random.random() < 0.1 and (time.time() - last_scream_time) > 15:
                scream_msg = f"IMPORTANT: Simulated scream detected at random position"
                alertBase(scream_msg)
                last_scream_time = time.time()

            time.sleep(1)
        except Exception as e:
            print("Error in simulated main loop:", e)
finally:
    client_socket.close()
    print("[SIM] Simulation ended, socket closed")
