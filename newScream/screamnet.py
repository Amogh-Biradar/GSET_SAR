import os
import torch
import torchaudio
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# 🔹 1. Custom Dataset
# ===============================
class ScreamDataset(Dataset):
    def __init__(self, scream_dir, not_scream_dir):
        self.data = []
        self.labels = []

        for file in os.listdir(scream_dir):
            if file.endswith(".wav"):
                waveform, _ = torchaudio.load(os.path.join(scream_dir, file))
                resampler = torchaudio.transforms.Resample(orig_freq=44100, new_freq=10000)
                # Apply resampling
                waveform = resampler(waveform)
                self.data.append(waveform)
                self.labels.append(1)  # 1 for scream

        for file in os.listdir(not_scream_dir):
            if file.endswith(".wav"):
                waveform, _ = torchaudio.load(os.path.join(not_scream_dir, file))
                resampler = torchaudio.transforms.Resample(orig_freq=44100, new_freq=10000)
                # Apply resampling
                waveform = resampler(waveform)
                self.data.append(waveform)
                self.labels.append(0)  # 0 for not scream

        combined = list(zip(self.data, self.labels))
        random.shuffle(combined)
        self.data, self.labels = zip(*combined)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        waveform = self.data[idx]
        
        label = self.labels[idx]
        waveform = waveform.mean(dim=0, keepdim=True)  # mono (1, N)
        waveform = F.pad(waveform, (0, 90703 - waveform.shape[1]))  # pad to 160000 samples
        return waveform, torch.tensor(label, dtype=torch.float32)

# ===============================
# 🔹 2. Updated Model Architecture
# ===============================
class ScreamClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(16, 64, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(64, 128, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(128, 256, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.AdaptiveAvgPool1d(128),  # 256
            nn.Flatten(),
            nn.Linear(256 * 128, 64), # 512, 256, 128
            nn.ReLU(),
            nn.Linear(64, 1), # 128
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# ===============================
# 🔹 3. Training Function
# ===============================
def train_model(model, train_loader, val_loader, epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.BCELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")
        torch.save(model.state_dict(), "scream_classifier.pth")


    # Save model
    print("Model saved to scream_classifier.pth")

# ===============================
# 🔹 4. Evaluation Function
# ===============================
def evaluate_model(model, test_loader):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            predictions = (outputs > 0.5).float().squeeze()

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Scream", "Scream"])
    disp.plot(cmap='Purples')
    plt.title("Confusion Matrix")
    plt.show()

    # Accuracy and F1
    accuracy = np.mean(np.array(y_true) == np.array(y_pred))
    f1 = f1_score(y_true, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"F1 Score: {f1:.4f}")

# ===============================
# 🔹 5. Main Script
# ===============================
if __name__ == "__main__":
    print('dataset')
    dataset = ScreamDataset("newData/Screaming", "newData/NotScreaming")
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    print('splitting')
    train_set, test_set = random_split(dataset, [train_size, val_size])

    print('loaders')
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=4)

    model = ScreamClassifier().to(device)

    print('training')
    #train_model(model, train_loader, test_loader, epochs=50)

    print('testing')

    model = ScreamClassifier().to(device)
    model.load_state_dict(torch.load("scream_classifier.pth", map_location=device))
    print("✅ Model loaded from scream_classifier.pth")

    evaluate_model(model, test_loader)
