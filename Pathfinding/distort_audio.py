import numpy as np
from scipy.io import wavfile
import os

def distort_audio(input_path, output_path, distortion_level=0.05):
    fs, data = wavfile.read(input_path)
    # Convert to float for processing
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32767
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483647
    # Add small random noise
    noise = np.random.normal(0, distortion_level, size=data.shape)
    distorted = data + noise
    # Apply a mild nonlinear distortion (tanh)
    distorted = np.tanh(distorted)
    # Normalize and convert back to int16
    distorted = distorted / np.max(np.abs(distorted))
    distorted = (distorted * 32767).astype(np.int16)
    wavfile.write(output_path, fs, distorted)
    print(f"Distorted file written to: {output_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    in_path = os.path.join(base_dir, "test_mic_a_0.2_shift.wav")
    out_path = os.path.join(base_dir, "distorted_test_mic_a_0.2_shift.wav")
    distort_audio(in_path, out_path, distortion_level=0.25)
