# import necessary libraries/files
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.optim as optim
import torchaudio
import os
from torch.utils.data import TensorDataset
import torch.nn.functional as F
from torch.utils.data import DataLoader
from filterModel import Denoiser, spec_to_wave

model = Denoiser()
model.load_state_dict(torch.load("denoiser_model.pth"))
model.eval()

clean_spec = torch.load('specs/clean/0.pt')
mixed = torch.load('specs/mixed/0.pt')

output = model(mixed.unsqueeze(0)).squeeze(0)

# Get the min shared shape
min_freq = min(output.shape[1], clean_spec.shape[1])
min_time = min(output.shape[2], clean_spec.shape[2])

# Crop both tensors to match
output = output[:, :min_freq, :min_time]
clean_spec = clean_spec[:, :min_freq, :min_time]

out_wave = spec_to_wave(output)
clean_wave = spec_to_wave(clean_spec)

print(type(out_wave), out_wave.shape)

torchaudio.save("filtered.wav", out_wave, 16000)
torchaudio.save("clean.wav", clean_wave, 16000)