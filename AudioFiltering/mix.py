# Import necessary libraries
import os
import random
import torchaudio
import torch

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
    return mixed, clean_wave, sr

# Mix the entire folder of clean and noisy audio files
def mixFolder():
    # Sort files in clean and noisy folders
    clean_files = sorted([f for f in os.listdir('cleanData/') if f.endswith('.wav')])
    drone_files = sorted([f for f in os.listdir('noisyData/') if f.endswith('.wav')])

    # For each file in the clean folder, take clean file and randomly select noisy file
    for i, clean_file in enumerate(clean_files):
        clean_path = os.path.join('cleanData/', clean_file)
        drone_path = os.path.join('noisyData/', random.choice(drone_files))

        # Select volume randomizer between -5 and 5 dB, then call mix function to mix files
        snr_db = random.randint(*(-5,5))
        mixed, clean, sr = mix(clean_path, drone_path, snr_db)

        # Save mixed and clean audio files to respective folders
        torchaudio.save(f"workingData/mixed/{i}.wav", mixed, sr)
        torchaudio.save(f"workingData/clean/{i}.wav", clean, sr)

mixFolder()