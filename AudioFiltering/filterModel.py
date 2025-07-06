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

# Initialize and create dataset and functions for spectrogram mixed and clean sounds 
class NoisyData(Dataset):
    # Initialize dataset and create ordered mixed and file arrays, directories, length of arrays
    def __init__(self, mixed_dir, clean_dir):
        self.mixed_files = sorted(os.listdir(mixed_dir))
        self.clean_files = sorted(os.listdir(clean_dir))
        self.mixed_dir = mixed_dir
        self.clean_dir = clean_dir
        self.length = len(self.mixed_files)

    # Get one mixed and clean sound and convert to spectrogram
    def oneSound(self, idx):
        mix_wave, _ = torchaudio.load(os.path.join(self.mixed_dir, self.mixed_files[idx]))
        clean_wave, _ = torchaudio.load(os.path.join(self.clean_dir, self.clean_files[idx]))

        mix_spec = wave_to_spec(mix_wave)
        clean_spec = spec_to_wave(clean_wave)

        return mix_spec, clean_spec
    
    # Get all items from the dataset and store in a mixed and clean spectrogram list
    def getItems(self):
        mixedSpecs = []
        cleanSpecs = []

        for idx in range(self.length):
            mixedSpecs.append(self.oneSound(idx)[0])
            cleanSpecs.append(self.oneSound(idx)[1])

        return mixedSpecs, cleanSpecs

# Denoiser neural network
class Denoiser(nn.Module):
    # Initialize CNN
    def __init__(self):
        super().__innit__()
        # Encoder layer to process and condense data
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU()
        )

        # Decoder layer to reconstruct data and generate output
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    # Function to generate output
    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)

def split():
    dataset = NoisyData("/data/mixed", "/data/clean")
    mixed = dataset.getItems()[0]
    clean = dataset.getItems()[1]

    trainMixed = mixed[int(len(mixed) * 0.2):]
    trainClean = clean[int(len(clean) * 0.2):]
    testMixed = mixed[:int(len(mixed) * 0.2)]
    testClean = clean[:int(len(clean) * 0.2)]

    trainSet = ListAudioDataset(trainMixed, trainClean)

    return trainSet, testMixed, testClean

def train(trainSet):
    # Data loader to load data in batches for model training
    loader = DataLoader(trainSet, batch_size=4, shuffle=True)

    # Initialize the model, loss function, and optimizer (maximize performance)
    model = Denoiser()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    # Number of training cycles
    num_epochs = 20
    for epoch in range(num_epochs):
        epoch_loss = 0.0  # Will the average loss of the model over the epoch

        for mix_spec, clean_spec in loader:
            # Denoise the mixed spectrogram to create cleaned version
            output = model(mix_spec)

            # Compute loss with Mean Squared Error by comparing output to original clean spectrogram
            loss = criterion(output, clean_spec)

            # Backpropagation by clearing old gradients, calculating new gradients, and updating weights of model parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate the loss for the epoch
            epoch_loss += loss.item()

        # Calculate and print the average loss per epoch
        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch + 1}/{num_epochs} - Avg Loss: {avg_loss:.4f}")

        # Save trained model to the project
        torch.save(model.state_dict(), "denoiser_model.pth")

if __name__ == "__main__":
    # Split spectrogram dataset into training set and validation clean and mixed sets
    trainSet, testMixed, testClean = split()

    # Train the model
    train(trainSet)

    