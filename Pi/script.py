'''
TO DO:
1. image detection
2. localization
3. gyroscope
4. base station sending and receiving
5. dealing with special commands
6. base station (receiving, sending, flying, special commands), then arduino code (flying)
'''

import sounddevice as sd
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import time
import torch.nn.functional as F
import time
import cv2
import soundfile as sf
import noisereduce as nr

# ====== CONFIG ======
DEVICE_IDS = [1, 2, 3]  # hardware IDs for USB mics (switch if necessary)
SAMPLE_RATE = 44100
RECORD_SECONDS = 2.267
FILTER_PATH = 'denoiser_model.pth'
SCREAM_PATH = 'screamnet_best.pth'

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
class ScreamNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(256, 512, 3, padding=1)  # New conv layer
        self.bn4 = nn.BatchNorm1d(512)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear((input_dim // 16) * 512, 128)  # Adjusted for extra pooling
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
    def forward(self, x):
        x = self.pool(self.bn1(torch.relu(self.conv1(x))))
        x = self.dropout(x)
        x = self.pool(self.bn2(torch.relu(self.conv2(x))))
        x = self.dropout(x)
        x = self.pool(self.bn3(torch.relu(self.conv3(x))))
        x = self.dropout(x)
        x = self.pool(self.bn4(torch.relu(self.conv4(x))))  # New conv layer
        x = self.dropout(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.dropout(x, 0.4, self.training)
        x = torch.relu(self.fc2(x))
        x = torch.dropout(x, 0.4, self.training)
        x = self.fc3(x)
        return x
    
# ====== AUDIO PROCESSING ======
def preprocess_wave(waveform):
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if waveform.shape[1] > 100000:
        waveform = waveform[:, :100000]
    else:
        waveform = F.pad(waveform, (0, 100000 - waveform.shape[1]))
    return waveform

# ====== LOAD MODELS ======
filterModel = Denoiser1D()
filterModel.load_state_dict(torch.load(FILTER_PATH, map_location='cpu'))
filterModel.eval()

screamModel = ScreamNet(input_dim=400000, num_classes=2)
screamModel.load_state_dict(torch.load('SCREAM_PATH', map_location='cpu'))
screamModel.eval()

# ====== AUDIO FUNCTION ======
def record_and_filter(device_id):
    print(f"[INFO] Recording from device {device_id}...")
    audio = sd.rec(int(RECORD_SECONDS * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32', device=device_id)
    sd.wait()
    audio = preprocess_wave(torch.tensor(audio))

    # === Denoise using 1D CNN ===
    with torch.no_grad():
        output_wave = filterModel(audio.unsqueeze(0)).squeeze(0) * 35

        # Convert to 1D numpy array for noise reduction
        output_np = output_wave.squeeze().detach().numpy()
        output_denoised = nr.reduce_noise(y=output_np, sr=44100, prop_decrease = 0.75)

        # Convert back to tensor for saving
        output_wave = torch.tensor(output_denoised, dtype=torch.float32).unsqueeze(0) * 50
    
    return output_wave

# ====== Scream Detection ======
def class_scream(wavIdx):
    with torch.no_grad():
        res1 = torch.argmax(screamModel(wavIdx[0].unsqueeze(0)), dim=1).item() == 1
        res2 = screamModel(wavIdx[1]) == "Screaming"
        res3 = screamModel(wavIdx[2]) == "Screaming"
    if res1 or res2 or res3:
        coords = (0, 0)
        alertBase(f"Human scream detected. Drone at coordinates: {coords}.")
        localize(res1, res2, res3, wavIdx)

# ====== Localization if Scream Detected ======
def localize(res1, res2, res3, wavIdx):
    # FFT to isolate scream and find that amplitude? or time dependent?
    heading = 1
    meters = 1
    alertBase(f"IMPORTANT: Localized scream. Move on heading: {heading} FOR meters: {meters}.")
    # if drone is overhead alert base saying person found at coords
    found()
    # visual detection

# ====== Procedure for Person Found ======
def found():
    image = takeImg()
    message = classImage(image)
    coords = 1
    if "person" in message:
        alertBase(f"IMPORTANT: VISUAL AND AUDIO CONFIRMATION. PERSON FOUND AT {coords}.")
    else:
        alertBase(f"IMPORTANT: AUDIO CONFIRMATION. PERSON FOUND AT {coords}. VISUAL INCONCLUSIVE.")
    sendImg(image)

# ====== Image Capture ======
def takeImg():
    cap = cv2.VideoCapture(0)  # 0 is the default camera index
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Error: Failed to capture image.")
        return None

    cv2.imwrite('capture.jpg', frame)
    return frame

# ====== Image Classification ======
def classImage(image):
    coords = 1
    return (f"Here class the image with mobilenet and potentially other network. Found at coords: {coords}.")

# ====== Detect Object with Ultrasonic ======
def ultrasonic():
    alerted = False
    running = True
    while running:
        distance = 1
        if distance < 300 and not alerted: # cm
            alertBase(f"ALERT: OBJECT DETECTED. DISTANCE: {distance}")
            alerted = True
        else:
            running = False
            alertBase(f"ALERT: OBJECT CLEAR.")

# ====== Stabilize with Gyroscope ======
def gyro():
    # read gyro + gather metric data
    alertBase(f"GYRO: ")

# ====== Alert Base Station ======
def alertBase(message):
    print(message)

# ====== Image Sending ======
def sendImg(image):
    print(image)

# ====== Receive from Base Station ======
# receive overrides and perform actions

x = 0
last_trigger_time = time.time()
interval = 31  # seconds
# ====== MAIN LOOP ======
while True:
    # Ultrasonic
    ultrasonic()
    # Gyroscope for flight stabilization
    gyro()

    # Image Processes
    current_time = time.time()
    if current_time - last_trigger_time >= interval:
        image = takeImg()
        message = classImage(image)
        alertBase(message)
        if "person" in message:
            coords = 1
            alertBase(f"IMPORTANT: PERSON SPOTTED AT COORDS: {coords}.")
            sendImg(image)
        if "signal flare" in message:
            coords = 1
            alertBase(f"IMPORTANT: FLARE SPOTTED AT COORDS: {coords}.")
            sendImg(image)
        last_trigger_time = current_time
    # Optional: sleep briefly to reduce CPU usage
    time.sleep(0.1)
    
    # Audio Capture
    if x == 0:
        wavIdx = []
        try:
            for idk, dev_id in enumerate(DEVICE_IDS):
                wavIdx.append(record_and_filter(dev_id)) # [1, N]

        except KeyboardInterrupt:
            print("Exiting...")
            break
        x += 1
    elif x < 4:
        try:
            for idx, dev_id in enumerate(DEVICE_IDS):
                wavIdx[idx] = torch.cat((wavIdx[idx], record_and_filter(dev_id)), dim=1)

        except KeyboardInterrupt:
            print("Exiting...")
            break
        x += 1
    else:
        for idx, dev_id in enumerate(DEVICE_IDS):
                class_scream(wavIdx)
        x = 0

    # Special Commands
    # Take + analyze image
    # Send image
    # Send audio sample?
    # Send coordinates
