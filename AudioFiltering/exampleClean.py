import torch
import torchaudio
import torch.nn.functional as F
import matplotlib.pyplot as plt
from filterModel import Denoiser1D  # your new 1D model
import soundfile as sf
import noisereduce as nr

# === Load Model ===
model = Denoiser1D()
model.load_state_dict(torch.load("denoiser_model.pth"))
model.eval()

# === Load .wav Files ===
clean_wave, sample_rate = torchaudio.load('workingData/clean/9.wav')
mixed_wave, _ = torchaudio.load('workingData/mixed/9.wav')

# === Convert to mono and fixed length (60000) ===
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
    
    # Convert to 1D numpy array for noise reduction
    output_np = output_wave.squeeze().detach().numpy()
    output_denoised = nr.reduce_noise(y=output_np, sr=sample_rate, prop_decrease = 0.75)

    # Convert back to tensor for saving
    #output_wave = torch.tensor(output_denoised, dtype=torch.float32).unsqueeze(0) * 50
    output_wave = torch.tensor(output_denoised, dtype=torch.float32)
    output_wave = output_wave / torch.max(torch.abs(output_wave))  # normalize to [-1, 1]
    output_wave = output_wave.unsqueeze(0)

# === Save Output Files ===
torchaudio.save("sampleAudio/filtered1.wav", output_wave, sample_rate)
torchaudio.save("sampleAudio/clean1.wav", clean_wave, sample_rate)
torchaudio.save("sampleAudio/mixed1.wav", mixed_wave, sample_rate)

# === Plot for Comparison ===
plt.figure(figsize=(12, 4))

plt.plot(mixed_wave[0].numpy(), label="Mixed with Drone", alpha=0.5)
plt.plot(clean_wave[0].numpy(), label="Original Clean", alpha=0.5)
plt.plot(output_wave[0].detach().numpy(), label="Neural Filter", alpha=0.7)
plt.legend()
plt.title("Waveform Comparison")
plt.show()
