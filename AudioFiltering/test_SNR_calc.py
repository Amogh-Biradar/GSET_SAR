from tylerEvaluation import calculate_SNR
import numpy as np
import noisereduce as nr
from scipy.io import wavfile

# Example: Load a noisy audio file and a clean reference file
# Replace these with your actual file paths
noisy_file = 'noisy.wav'      # Path to your noisy audio file
clean_file = 'clean.wav'      # Path to your clean (ground truth) audio file

# Read audio files
rate_noisy, noisy = wavfile.read(noisy_file)
rate_clean, clean = wavfile.read(clean_file)

# Ensure both signals are the same length
min_len = min(len(noisy), len(clean))
noisy = noisy[:min_len]
clean = clean[:min_len]

# Calculate SNR before noise reduction
noise = noisy - clean
snr_before = calculate_SNR(clean, noise)

# Apply noise reduction
reduced = nr.reduce_noise(y=noisy, sr=rate_noisy)

# Calculate SNR after noise reduction
noise_after = reduced - clean
snr_after = calculate_SNR(clean, noise_after)

print(f"SNR before noise reduction: {snr_before:.2f} dB")
print(f"SNR after noise reduction: {snr_after:.2f} dB")
print(f"Improvement: {snr_after - snr_before:.2f} dB")

