# import necessary libraries
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.optim as optim
import torchaudio
import os
from torch.utils.data import TensorDataset
import torch.nn.functional as F
from torch.utils.data import DataLoader


# Raw waveform sound to spectrogram for analysis
def wave_to_spec(waveform):
    # Check to see if waveform is none. Otherwise, convert to spectrogram
    if waveform is None:
        print(f"Skipping file due to load failure.")
        return None  # or skip, or handle gracefully
    else:
        # Crop/pad input waveform to fixed length before spectrogram
        fixed_length = 60000  # at 16000 Hz
        if waveform.shape[1] > fixed_length:
            waveform =  waveform[:, :fixed_length]
        else:
            padding = fixed_length - waveform.shape[1]
            waveform =  torch.nn.functional.pad(waveform, (0, padding))

        # Convert to mono for accuract spectrogram conversion
        if waveform.ndim == 2 and waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Ensure correct two-dimesnional shape
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        # Converts to spectrogram
        # n_fft = size of FFT (data bins)
        # hop_length = number of samples between successive frames (overlap)
        spectrogram = torchaudio.transforms.Spectrogram(n_fft=512, hop_length=256, power=2)(waveform)

        # Add log scaling and normalization
        spec_db = 10 * torch.log10(spectrogram + 1e-10)  # convert power to dB
        epsilon = 1e-8
        spec_db_norm = (spec_db - spec_db.min()) / (spec_db.max() - spec_db.min() + epsilon)
        print('Successfully converted waveform to spectrogram.')
        return spec_db_norm

# Converts processed spectrogram back to waveform
def spec_to_wave(spec):
    # Ensure shape is [1, freq, time] for GriffinLim
    if spec.ndim == 2:
        spec = spec.unsqueeze(0)

    # Error check for malformed spectrogram
    if spec.size(1) != (512 // 2 + 1):
        raise ValueError(f"Expected frequency bins = {512 // 2 + 1}, got {spec.size(1)}")
 


    # Inverts spectrogram transformation and converts back to linear scale
    inverse_transform = torchaudio.transforms.GriffinLim(n_fft=512, hop_length=256)(spec)
    return inverse_transform

# Initialize and create dataset and functions for spectrogram mixed and clean sounds 
class NoisyData(Dataset):
    # Initialize dataset and create ordered mixed and file arrays, directories, length of arrays
    def __init__(self, mixed_dir, clean_dir):
        # Example: get list of only .wav files
        self.mixed_files = [f for f in os.listdir(mixed_dir) if f.endswith(".wav")]
        self.clean_files = [f for f in os.listdir(clean_dir) if f.endswith(".wav")]
        self.mixed_dir = mixed_dir
        self.clean_dir = clean_dir
        self.length = len(self.mixed_files)

    # Safely load files
    def safe_load(self, path):
        # Try to open file with torchaudio
        try:
            return torchaudio.load(path)
        # If impossible, print warning and return None
        except Exception as e:
            print(f"Warning: could not load {path}: {e}")
            return None, None

    # Get one mixed and clean sound and convert to spectrogram
    def oneSound(self, idx):
        mix_wave, _ = self.safe_load(os.path.join(self.mixed_dir, self.mixed_files[idx]))
        clean_wave, _ = self.safe_load(os.path.join(self.clean_dir, self.clean_files[idx]))

        mix_spec = wave_to_spec(mix_wave)
        clean_spec = wave_to_spec(clean_wave)
        if mix_spec != None and clean_spec != None:
            torch.save(mix_spec, f"specs/mixed/{idx}.pt")
            torch.save(clean_spec, f"specs/clean/{idx}.pt")

    # Get all items from the dataset and store in a mixed and clean spectrogram list
    def saveItems(self):
        for idx in range(1200):
            self.oneSound(idx)

# Denoiser neural network
class Denoiser(nn.Module):
    # Initialize CNN
    def __init__(self):
        super().__init__()
        # Encoder layer to process and condense data
        self.encoder = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
        nn.InstanceNorm2d(16),
        nn.LeakyReLU(0.1),
        nn.Dropout(0.05),

        nn.Conv2d(16, 64, kernel_size=3, stride=2, padding=1),
        nn.InstanceNorm2d(64),
        nn.LeakyReLU(0.1),
        nn.Dropout(0.1),

        nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1),
        nn.InstanceNorm2d(256),
        nn.LeakyReLU(0.1),
        nn.Dropout(0.15),

        nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
        nn.InstanceNorm2d(512),
        nn.LeakyReLU(0.1),
        nn.Dropout(0.2)
    )

        # Decoder layer to reconstruct the spectrogram from the encoded data
        self.decoder = nn.Sequential(
        nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
        nn.InstanceNorm2d(256),
        nn.LeakyReLU(0.1),
        nn.Dropout(0.15),

        nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1),
        nn.InstanceNorm2d(64),
        nn.LeakyReLU(0.1),
        nn.Dropout(0.1),

        nn.ConvTranspose2d(64, 16, kernel_size=4, stride=2, padding=1),
        nn.InstanceNorm2d(16),
        nn.LeakyReLU(0.1),
        nn.Dropout(0.05),

        nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
        nn.Sigmoid()
    )
    
    # Function to generate output
    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)

# Normalize spectrogram length with padding
def pad_spectrograms(spec_list):
    max_len = max(spec.shape[-1] for spec in spec_list)
    padded = [
        F.pad(spec, (0, max_len - spec.shape[-1]))  # pad right side of time dim
        for spec in spec_list
    ]
    return torch.stack(padded)

def split():
    # Get spectrogram files and sort into training and testing lists
    mixed = sorted([f for f in os.listdir('specs/mixed') if f.endswith('.pt')])

    trainMixedNames = mixed[int(len(mixed) * 0.2):]
    trainMixed = []
    trainClean = []
    testMixedNames = mixed[:int(len(mixed) * 0.2)]
    testMixed = []
    testClean = []

    for name in trainMixedNames:
        # Load spectrograms from files and append to training list
        spec = torch.load(os.path.join('specs/mixed', name))
        trainMixed.append(spec)

        spec = torch.load(os.path.join('specs/clean', name))
        trainClean.append(spec)
    for name in testMixedNames:
        # Load spectrograms from files and append to testing list
        spec = torch.load(os.path.join('specs/mixed', name))
        testMixed.append(spec)

        spec = torch.load(os.path.join('specs/clean', name))
        testClean.append(spec)

    # Convert lists to tensors for machine learning and normalizes length
    trainMixedTensor = pad_spectrograms(trainMixed)
    trainCleanTensor = pad_spectrograms(trainClean)

    # Create dataset
    trainSet = TensorDataset(trainMixedTensor, trainCleanTensor)

    return trainSet, testMixed, testClean

def train(trainSet):
    # Data loader to load data in batches for model training
    loader = DataLoader(trainSet, batch_size=4, shuffle=True)

    # Initialize the model, loss function, optimizer and scheduler (maximize performance)
    model = Denoiser()
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # Training loop
    # Number of training cycles
    num_epochs = 35
    for epoch in range(num_epochs):
        epoch_loss = 0.0  # Will the average loss of the model over the epoch

        for mix_spec, clean_spec in loader:
            # Denoise the mixed spectrogram to create cleaned version
            output = model(mix_spec)

            # Crop both tensors to same size for accurate loss calculation
            min_len = min(output.shape[-1], clean_spec.shape[-1])
            output = output[..., :min_len]
            clean_spec = clean_spec[..., :min_len]

            min_freq = min(output.shape[2], clean_spec.shape[2])
            output = output[:, :, :min_freq, :]
            clean_spec = clean_spec[:, :, :min_freq, :]

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

        # Adjust learning rate to counter plateaus in training
        scheduler.step(avg_loss)

        # Save trained model to the project
        torch.save(model.state_dict(), "denoiser_model.pth")

# Calculate Mean Squared Error by comparing actual signal to model output
def calcMSE(true_signal: torch.Tensor, cleaned_signal: torch.Tensor) -> float:
    # Flatten both tensors
    true_signal = true_signal.flatten()
    cleaned_signal = cleaned_signal.flatten()

    if true_signal.shape != cleaned_signal.shape:
        raise ValueError("Signal and noise must have the same shape.")

    mse = torch.mean((true_signal - cleaned_signal) ** 2).item()
    return mse

# Calculate accuracy of the model by comparing the true signal to the cleaned signal
def calcAccuracy(true_signal: torch.Tensor, cleaned_signal: torch.Tensor) -> float:
    true_signal = true_signal.flatten()
    power = torch.mean(true_signal ** 2).item()
    if power < 1e-8:
        return 0.0  # signal power too small, consider accuracy zero
    
    mse = calcMSE(true_signal, cleaned_signal)
    accuracy = 1 - (mse / power)
    return max(0.0, accuracy)  # clamp negative values to 0

# Test the model
def test(testMixed, testClean):
    # Specify model architecture, load trained weights, and set to evaluation mode
    model = Denoiser()
    model.load_state_dict(torch.load("denoiser_model.pth"))
    model.eval()

    # Create an accuracy sum and number of tests for average accuracy calculation
    sumAccuracy = 0
    numTests = len(testMixed)

    # Iterate through test dataset; filter each mixed sound and calculate accuracy
    for idx in range(len(testMixed)):
        clean_spec = testClean[idx]
        output = model(testMixed[idx].unsqueeze(0)).squeeze(0)

        # Print tensor shapes for debugging
        # print("Pre output shape:", output.shape)
        # print("Pre clean shape:", clean_spec.shape)

        # Get the min shared shape
        min_freq = min(output.shape[1], clean_spec.shape[1])
        min_time = min(output.shape[2], clean_spec.shape[2])

        # Crop both tensors to match
        output = output[:, :min_freq, :min_time]
        clean_spec = clean_spec[:, :min_freq, :min_time]

        # Print tensor shapes for debugging
        # print("Done output shape:", output.shape)
        # print("Done clean shape:", clean_spec.shape)

        # Convert back to waves, then cut to same length - for .wav comparison, which is not as good
        # out_wave = spec_to_wave(output)
        # clean_wave = spec_to_wave(clean_spec)
        # min_len = min(clean_wave.shape[-1], out_wave.shape[-1])
        # clean_wave = clean_wave[..., :min_len]
        # out_wave = out_wave[..., :min_len]

        sumAccuracy += calcAccuracy(clean_spec.flatten(), output.flatten()) * 100

    # Print average test accuracy
    print(f"Average Accuracy: {sumAccuracy / numTests:.2f}%")
    print("Testing complete.")


if __name__ == "__main__":
    # Create dataset and save spectrograms; comment after first run
    dataset = NoisyData("workingData/mixed", "workingData/clean")
    dataset.saveItems()

    # Split spectrogram dataset into training set and validation clean and mixed sets
    trainSet, testMixed, testClean = split()

    # Train the model
    train(trainSet)

    # Test them model
    test(testMixed, testClean)

