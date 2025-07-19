# Import necessary libraries
import os
import random
import torchaudio
import torch
import torch.nn.functional as F
import soundfile as sf
import noisereduce as nr
import torch.nn as nn

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

# Function to mix individual clean and noisy files
def mix(clean_path, drone_path, snr_db=0):
    # Load clean and noisy audio files
    clean_wave, sr = torchaudio.load(clean_path)
    drone_wave, _ = torchaudio.load(drone_path)
    # Multiply length of drone segment by 50 to ensure adequate length
    drone_wave = torch.cat((drone_wave, drone_wave, drone_wave, drone_wave, drone_wave, drone_wave, drone_wave, drone_wave, drone_wave, drone_wave,
                            drone_wave, drone_wave, drone_wave, drone_wave, drone_wave, drone_wave, drone_wave, drone_wave, drone_wave, drone_wave,
                            drone_wave, drone_wave, drone_wave, drone_wave, drone_wave, drone_wave, drone_wave, drone_wave, drone_wave, drone_wave,
                            drone_wave, drone_wave, drone_wave, drone_wave, drone_wave, drone_wave, drone_wave, drone_wave, drone_wave, drone_wave,
                            drone_wave, drone_wave, drone_wave, drone_wave, drone_wave, drone_wave, drone_wave, drone_wave, drone_wave, drone_wave,
                            drone_wave, drone_wave, drone_wave, drone_wave, drone_wave, drone_wave, drone_wave, drone_wave, drone_wave, drone_wave,
                            drone_wave, drone_wave, drone_wave, drone_wave, drone_wave, drone_wave, drone_wave, drone_wave, drone_wave, drone_wave), dim=1)

    # Match length and shorten audio files
    min_len = min(clean_wave.size(1), drone_wave.size(1))
    clean_wave = clean_wave[:, :min_len]
    drone_wave = drone_wave[:, :min_len]

    # Scale drone to match SNR, for randomized drone to clean volume ratio and dataset variability
    clean_power = clean_wave.pow(2).mean()
    drone_power = drone_wave.pow(2).mean()
    snr_linear = 10 ** (snr_db / 10)
    scale = (clean_power / (snr_linear * drone_power)).sqrt()
    drone_scaled = drone_wave * scale

    # Mix and return the audio files
    mixed = clean_wave + drone_scaled
    return mixed, sr

# Mix the entire folder of clean and noisy audio files
def mixFolder(outFolder):
    # Sort files in clean and noisy folders
    clean_files = sorted([f for f in os.listdir(f"data/{outFolder}/") if f.endswith('.wav')])
    drone_files = sorted([f for f in os.listdir('noisyData/') if f.endswith('.wav')])

    # For each file in the clean folder, take clean file and randomly select noisy file
    for i, clean_file in enumerate(clean_files):
        clean_path = os.path.join(f"data/{outFolder}/", clean_file)
        drone_path = os.path.join('noisyData/', random.choice(drone_files))

        # Select volume randomizer between -5 and 5 dB, then call mix function to mix files
        snr_db = random.randint(*(-5,5))
        mixed, sr = mix(clean_path, drone_path, snr_db)

        model = Denoiser1D()
        model.load_state_dict(torch.load("denoiser_model.pth"))
        model.eval()

        def preprocess_wave(waveform):
            waveArray = []
            for i in range(5):
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                if waveform.shape[1] > 400000:
                    waveform = waveform[:, :400000]
                    waveArray.append(waveform)
                    waveform = waveform[:, 400000:]
                else:
                    waveform = F.pad(waveform, (0, 400000 - waveform.shape[1]))
                    waveArray.append(waveform)
            return waveArray

        mixed = preprocess_wave(mixed)

        with torch.no_grad():
                fa = model(mixed[0].unsqueeze(0)).squeeze(0)
                
                # Convert to 1D numpy array for noise reduction
                output_np = fa.squeeze().detach().numpy()
                output_denoised = nr.reduce_noise(y=output_np, sr=sr, prop_decrease = 0.75)

                # Convert back to tensor for saving
                #output_wave = torch.tensor(output_denoised, dtype=torch.float32).unsqueeze(0) * 50
                fa = torch.tensor(output_denoised, dtype=torch.float32)
                fa = fa / torch.max(torch.abs(fa))  # normalize to [-1, 1]
                fa = fa.unsqueeze(0)

        # for i in range(4):
        #     # === Denoise using 1D CNN ===
        #     with torch.no_grad():
        #         output_wave = model(mixed[i+1].unsqueeze(0)).squeeze(0)
                
        #         # Convert to 1D numpy array for noise reduction
        #         output_np = output_wave.squeeze().detach().numpy()
        #         output_denoised = nr.reduce_noise(y=output_np, sr=sr, prop_decrease = 0.75)

        #         # Convert back to tensor for saving
        #         #output_wave = torch.tensor(output_denoised, dtype=torch.float32).unsqueeze(0) * 50
        #         output_wave = torch.tensor(output_denoised, dtype=torch.float32)
        #         output_wave = output_wave / torch.max(torch.abs(output_wave))  # normalize to [-1, 1]
        #         output_wave = output_wave.unsqueeze(0)
            
        #     torch.cat((fa, output_wave), dim=0)


        # Save mixed and clean audio files to respective folders
        torchaudio.save(f"newData/{outFolder}/{i}.wav", fa, sr)

mixFolder('Screaming')
mixFolder('NotScreaming')