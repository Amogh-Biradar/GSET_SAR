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


# Initialize and create dataset and functions for spectrogram mixed and clean sounds 
# class NoisyData(Dataset):
#     # Initialize dataset and create ordered mixed and file arrays, directories, length of arrays
#     def __init__(self, mixed_dir, clean_dir):
#         # Example: get list of only .wav files
#         self.mixed_files = [f for f in os.listdir(mixed_dir) if f.endswith(".wav")]
#         self.clean_files = [f for f in os.listdir(clean_dir) if f.endswith(".wav")]
#         self.mixed_dir = mixed_dir
#         self.clean_dir = clean_dir
#         self.length = len(self.mixed_files)

#     # Safely load files
#     def safe_load(self, path):
#         # Try to open file with torchaudio
#         try:
#             return torchaudio.load(path)
#         # If impossible, print warning and return None
#         except Exception as e:
#             print(f"Warning: could not load {path}: {e}")
#             return None, None

#     # Get one mixed and clean sound and convert to spectrogram
#     def oneSound(self, idx):
#         mix_wave, _ = self.safe_load(os.path.join(self.mixed_dir, self.mixed_files[idx]))
#         clean_wave, _ = self.safe_load(os.path.join(self.clean_dir, self.clean_files[idx]))

#         mix_spec = wave_to_spec(mix_wave)
#         clean_spec = wave_to_spec(clean_wave)
#         if mix_spec != None and clean_spec != None:
#             torch.save(mix_spec, f"specs/mixed/{self.mixed_files[idx]}.pt")
#             torch.save(clean_spec, f"specs/clean/{self.clean_files[idx]}.pt")

    # Get all items from the dataset and store in a mixed and clean spectrogram list
    # def saveItems(self):
    #     for idx in range(1200):
    #         self.oneSound(idx)

    
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

def split():
    mixed_dir = 'workingData/mixed'
    clean_dir = 'workingData/clean'
    
    files = sorted([f for f in os.listdir(mixed_dir) if f.endswith('.wav')])
    files = files[:1000]
    split_idx = int(len(files) * 0.2)
    
    test_names = files[:split_idx]
    train_names = files[split_idx:]

    def load_wave(filename, directory):
        waveform, sr = torchaudio.load(os.path.join(directory, filename))
        waveform = waveform / (waveform.abs().max() + 1e-8)
        # Mono and padding/cropping]
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if waveform.shape[1] > 100000:
            waveform = waveform[:, :100000]
        else:
            waveform = F.pad(waveform, (0, 100000 - waveform.shape[1]))
        return waveform
    print("creating names")
    # === Training Set ===
    train_mixed = []
    train_clean = []
    for name in train_names:
        train_mixed.append(load_wave(name, mixed_dir))
        train_clean.append(load_wave(name, clean_dir))
    print("converting tensors")
    train_mixed_tensor = torch.stack(train_mixed)
    train_clean_tensor = torch.stack(train_clean)
    train_set = TensorDataset(train_mixed_tensor, train_clean_tensor)
    print("train set created")
    # === Test Set ===
    test_mixed = [load_wave(name, mixed_dir) for name in test_names]
    test_clean = [load_wave(name, clean_dir) for name in test_names]
    print("split done")
    return train_set, test_mixed, test_clean

def si_snr_loss(estimation, target, eps=1e-8):
        # Flatten both tensors (any shape → 1D)
    estimation = estimation.view(-1) - estimation.view(-1).mean()
    target = target.view(-1) - target.view(-1).mean()

    dot = torch.dot(estimation, target)
    target_energy = torch.sum(target ** 2) + eps

    proj = (dot / target_energy) * target
    noise = estimation - proj

    ratio = torch.sum(proj ** 2) / (torch.sum(noise ** 2) + eps)
    ratio = torch.clamp(ratio, min=1e-8)  # avoid log(0)
    si_snr = 10 * torch.log10(ratio)

    return -si_snr
# Combined spectrogram + waveform loss
def combined_loss(output_spec, target_spec):
    # === Spectrogram Loss ===
    spec_loss = F.l1_loss(output_spec, target_spec)

    # === Griffin-Lim Inversion ===
    def safe_wave(spec):
        # Clamp values and run Griffin-Lim safely
        spec = torch.clamp(spec, 0.0, 1.0)
        waveform = torchaudio.transforms.GriffinLim(
            n_fft=512,
            hop_length=256,
            win_length=512,
            power=1.0,
            n_iter=16,         # lower iteration for speed during training
        )(spec)
        return waveform

    # Invert to waveform (batch size 1 assumed)
    try:
        est_wave = safe_wave(output_spec.squeeze(1))
        target_wave = safe_wave(target_spec.squeeze(1))
    except Exception as e:
        print("Griffin-Lim inversion error:", e)
        return spec_loss  # fallback

    # Match lengths
    min_len = min(est_wave.shape[-1], target_wave.shape[-1])
    est_wave = est_wave[..., :min_len]
    target_wave = target_wave[..., :min_len]

    # === SI-SNR Loss ===
    if target_wave.abs().max() < 1e-4:
        snr_loss = torch.tensor(0.0)  # Avoid zero-division
    else:
        snr_loss = si_snr_loss(est_wave.squeeze(), target_wave.squeeze())

    # === Final Loss ===
    total_loss = 0.95 * spec_loss + 0.05 * snr_loss
    return total_loss


def train(trainSet):
    # Data loader to load data in batches for model training
    loader = DataLoader(trainSet, batch_size=4, shuffle=True)

    # Initialize the model, loss function, optimizer and scheduler (maximize performance)
    model = Denoiser1D()
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    print('parameters optimized')
    # Training loop
    # Number of training cycles
    num_epochs = 35
    for epoch in range(num_epochs):
        epoch_loss = 0.0  # Will the average loss of the model over the epoch

        for mix_wave, clean_wave in loader:
            output = model(mix_wave)
            
            min_len = min(output.shape[-1], clean_wave.shape[-1])
            output = output[..., :min_len]
            clean_wave = clean_wave[..., :min_len]

            loss = 0.2 * (si_snr_loss(output, clean_wave)) + 0.8 * F.l1_loss(output, clean_wave)

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

# Test the model
def test(testMixed, testClean):
    model = Denoiser1D()
    model.load_state_dict(torch.load("denoiser_model.pth"))
    model.eval()

    total_accuracy = 0.0
    numTests = len(testMixed)

    for mix_wave, clean_wave in zip(testMixed, testClean):
        with torch.no_grad():
            output = model(mix_wave.unsqueeze(0)).squeeze(0)  # [1, 1, T] → [1, T]

        min_len = min(output.shape[-1], clean_wave.shape[-1])
        output = output[..., :min_len]
        clean_wave = clean_wave[..., :min_len]

        # Compute SI-SNR loss
        snr_loss_val = si_snr_loss(output.squeeze(), clean_wave.squeeze()).item()
        si_snr_db = -snr_loss_val
        snr_acc = max(0.0, min((si_snr_db + 10) / 40, 1.0))  # map -10–30dB to 0–1

        # Compute L1 loss accuracy (lower loss = higher accuracy)
        l1 = F.l1_loss(output, clean_wave).item()
        l1_acc = max(0.0, min(1.0 - l1 / 0.5, 1.0))  # assume 0.5 is worst-case L1 loss

        # Combine
        acc = 0.2 * snr_acc + 0.8 * l1_acc
        total_accuracy += acc

    avg_accuracy = (total_accuracy / numTests) * 100
    print(f"Average Accuracy: {avg_accuracy:.2f}%")
    print("Testing complete.")


if __name__ == "__main__":
    # Create dataset and save spectrograms; comment after first run - outdated code (switched to .wav)
    # dataset = NoisyData("workingData/mixed", "workingData/clean")
    # dataset.saveItems()

    # Split spectrogram dataset into training set and validation clean and mixed sets
    trainSet, testMixed, testClean = split()

    # Train the model
    # train(trainSet)

    # Test them model
    test(testMixed, testClean)

