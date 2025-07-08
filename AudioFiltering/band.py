import numpy as np
from scipy.io import wavfile
from scipy.signal import iirnotch, butter, filtfilt
import os
def drone_noise_bandstopfilter(audio_recording_path, output_path):
    # 1. Read WAV file
    fs, data = wavfile.read(audio_recording_path)

    # Ensure data is float32 for processing
    data = data.astype(np.float32)

    # If stereo, convert to mono
    if data.ndim > 1:
        data = data.mean(axis=1)

    # 2. Apply notch filter at 130 Hz
    f0 = 130.0
    Q = 30.0
    b_notch, a_notch = iirnotch(f0, Q, fs)
    data = filtfilt(b_notch, a_notch, data)

    # 3. Apply bandstop filter for 150–6000 Hz
    low_cut = 150.0
    high_cut = 6000.0
    order = 4
    b_bandstop, a_bandstop = butter(order, [low_cut, high_cut], btype='bandstop', fs=fs)
    data = filtfilt(b_bandstop, a_bandstop, data)

    # 4. Normalize and convert to int16 to prevent clipping
    max_val = np.max(np.abs(data))
    if max_val > 0:
        data = data / max_val  # normalize to [-1, 1]
    data = (data * 32767).astype(np.int16)

    # 5. Save filtered WAV
    wavfile.write(output_path, fs, data)
    print("saved filtered audio to", output_path)
    
for file in os.listdir("workingData/mixed/"):
    if file.endswith(".wav"):
        audio_recording_path = os.path.join("workingData/mixed/", file)
        drone_noise_bandstopfilter(audio_recording_path, os.path.join("workingData/banded/", file))