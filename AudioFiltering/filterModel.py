# import necessary libraries
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.optim as optim
import torchaudio
import os

# Raw waveform sound to spectrogram for analysis
def wave_to_spec(waveform):
    # Converts to spectrogram
    # n_fft = size of FFT (data bins)
    # hop_length = number of samples between successive frames (overlap)
    # power = 2: energy, superior to amplitude
    # log10: to compress the range of values
    spectrogram = torchaudio.transforms.Spectrogram(n_fft=512, hop_length=256, power=2)(waveform)
    return spectrogram.log10()

# Converts processed spectrogram back to waveform
def spec_to_wave(spectrogram):
    # Inverts spectrogram transformation and converts back to linear scale
    inverse_transform = torchaudio.transforms.InverseSpectrogram(n_fft=512, hop_length=256)
    return inverse_transform(spectrogram.exp())

