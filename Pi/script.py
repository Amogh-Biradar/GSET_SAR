'''
TO DO:
1. localization + get coords/instructions
2. gyroscope, sensors, update coords/instructions
3. base station sending and interfacing with user
4. drone assembly, desoldering, weight testing
5. paper
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
RECORD_SECONDS = 9.072
FILTER_PATH = 'denoiser_model.pth'
SCREAM_PATH = 'scream_classifier.pth'

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
    
# ====== AUDIO PROCESSING ======
def preprocess_wave(waveform):
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
def record_and_filter(device_id):
    print(f"[INFO] Recording from device {device_id}...")
    audio = sd.rec(int(RECORD_SECONDS * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32', device=device_id)
    sd.wait()
    audio = preprocess_wave(torch.tensor(audio))

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
                alertBase(f"Human scream detected ({pred_raw * 100:.2f}% confidence). Drone at coordinates: {getCoords()}.")
            if pred_raw > 0.5:
                res[i] = True
            conf[i] = pred_raw
    

    if res[0] or res[1] or res[2]:
        localize(res, conf, wavIdx)

# ====== Localization if Scream Detected ======
def localize(res, conf, wavIdx):
    # FFT to isolate scream and find that amplitude? or time dependent?
    heading = 1
    meters = 1
    alertBase(f"IMPORTANT: Localized scream. Move on heading: {heading} FOR meters: {meters}.")
    # if drone is overhead alert base saying person found at coords
    found()
    # visual detection

# ====== Procedure for Person Found ======
def found():
    # image = takeImg()
    # message = classImage(image)
    # if "person" in message:
    #     alertBase(f"IMPORTANT: VISUAL AND AUDIO CONFIRMATION. PERSON FOUND AT {getCoords()}.")
    # else:
    alertBase(f"IMPORTANT: AUDIO CONFIRMATION. PERSON FOUND AT {getCoords()}. VISUAL INCONCLUSIVE.")
    # sendImg(image)

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
def gyro():
    # read gyro + gather metric data
    alertBase(f"GYRO: ")

# ====== Alert Base Station ======
def alertBase(message):
    print(message)

# # ====== Image Sending ======
# def sendImg(image):
#     print(image)

# # ====== Audio Sending ======
# def sendAudio(audio):
#     return

# ====== Get Coordinates ======
def getCoords():
    return 1

# # ====== Receive from Base Station ======
# def receiveBase():
#     return

x = 0
last_trigger_time = time.time()
interval = 31  # seconds
# ====== MAIN LOOP ======
while True:
    # Gyroscope for flight stabilization
    gyro()

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
    wavIdx = []
    try:
        for idk, dev_id in enumerate(DEVICE_IDS):
            wavIdx.append(record_and_filter(dev_id)) # [1, N]

            class_scream(wavIdx)

    except KeyboardInterrupt:
        print("Exiting...")
        break

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
