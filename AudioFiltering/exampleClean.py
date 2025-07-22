import torch
import torchaudio
import torch.nn.functional as F
import matplotlib.pyplot as plt
from filterModel import Denoiser1D  # your new 1D model
import soundfile as sf
import noisereduce as nr
import numpy as np

# === Load Model ===
model = Denoiser1D()
model.load_state_dict(torch.load("denoiser_model.pth"))
model.eval()

# === Load .wav Files ===
clean_wave, sample_rate = torchaudio.load('sampleAudio/clean1.wav')
mixed_wave, _ = torchaudio.load('sampleAudio/mixed1.wav')

# === Convert to mono and fixed length (100000 samples) ===
def preprocess_wave(waveform):
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if waveform.shape[1] > 100000:
        waveform = waveform[:, :100000]
    else:
        waveform = F.pad(waveform, (0, 100000 - waveform.shape[1]))
    return waveform

clean_wave = preprocess_wave(clean_wave)
mixed_wave = preprocess_wave(mixed_wave)

# === Denoise using 1D CNN ===
with torch.no_grad():
    output_wave = model(mixed_wave.unsqueeze(0)).squeeze(0)
    output_np = output_wave.squeeze().detach().numpy()

    # Post-process with noise reduction
    output_denoised = nr.reduce_noise(y=output_np, sr=sample_rate, prop_decrease=0.75)

    output_wave = torch.tensor(output_denoised, dtype=torch.float32)
    output_wave = output_wave / torch.max(torch.abs(output_wave))  # normalize
    output_wave = output_wave.unsqueeze(0)

# === Save Output Files ===
torchaudio.save("sampleAudio/filtered1.wav", output_wave, sample_rate)
torchaudio.save("sampleAudio/clean1.wav", clean_wave, sample_rate)
torchaudio.save("sampleAudio/mixed1.wav", mixed_wave, sample_rate)

# === Plot Waveform Comparison ===
plt.figure(figsize=(12, 4))
plt.plot(mixed_wave[0].numpy(), label="Mixed with Drone", alpha=0.5)
plt.plot(clean_wave[0].numpy(), label="Original Clean", alpha=0.5)
plt.plot(output_wave[0].numpy(), label="Neural Filtered", alpha=0.7)
plt.title("Waveform Comparison")
plt.legend()
plt.tight_layout()
plt.show()

# === Plot FFT of Removed Noise ===
from scipy.signal import savgol_filter

def compute_fft(signal, sr):
    freqs = np.fft.rfftfreq(len(signal), d=1/sr)
    fft_magnitude = np.abs(np.fft.rfft(signal))
    return freqs, fft_magnitude

def plot_fft_difference(clean_signal, filtered_signal, sr, smooth_window=101, polyorder=3):
    freqs, clean_fft = compute_fft(clean_signal, sr)
    _, filtered_fft = compute_fft(filtered_signal, sr)

    # Calculate delta magnitude in dB
    delta_mag = 20 * np.log10(np.abs(clean_fft - filtered_fft) + 1e-8)

    # Apply Savitzky–Golay smoothing
    smoothed_delta = savgol_filter(delta_mag, window_length=smooth_window, polyorder=polyorder)


    # Plot both raw and smoothed curves
    plt.figure(figsize=(12, 5))
    plt.plot(freqs, delta_mag, label='Raw |Clean| - |Filtered| (dB)', alpha=0.3, color='violet')
    plt.plot(freqs, smoothed_delta, label='Smoothed Difference', color='purple', linewidth=2)
    plt.title("FFT Magnitude Difference (Clean - Filtered)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude Difference (dB)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

# === Use it ===
plot_fft_difference(clean_wave[0].numpy(), output_wave[0].numpy(), sample_rate)
