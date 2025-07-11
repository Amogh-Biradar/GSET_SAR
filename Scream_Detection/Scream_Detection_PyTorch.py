# Scream_Detection_PyTorch.py
# PyTorch version of the scream detection model for easy model saving and download

import os
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Dataset class for PyTorch
class AudioDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Model definition (similar to your Keras model)
class ScreamNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(256, 512, 3, padding=1)  # New conv layer
        self.bn4 = nn.BatchNorm1d(512)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear((input_dim // 16) * 512, 128)  # Adjusted for extra pooling
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
    def forward(self, x):
        x = self.pool(self.bn1(torch.relu(self.conv1(x))))
        x = self.dropout(x)
        x = self.pool(self.bn2(torch.relu(self.conv2(x))))
        x = self.dropout(x)
        x = self.pool(self.bn3(torch.relu(self.conv3(x))))
        x = self.dropout(x)
        x = self.pool(self.bn4(torch.relu(self.conv4(x))))  # New conv layer
        x = self.dropout(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.dropout(x, 0.4, self.training)
        x = torch.relu(self.fc2(x))
        x = torch.dropout(x, 0.4, self.training)
        x = self.fc3(x)
        return x

# Data loading and preprocessing (same as before)
def load_data(data_path, metadata_path):
    features = []
    labels = []
    metadata = pd.read_csv(metadata_path)
    for index, row in metadata.iterrows():
        file_path = row['filepath']
        target_sr = 44100
        print("Processing file:", file_path)  # Debug: print file being processed
        audio, sample_rate = librosa.load(file_path, sr=target_sr)
        mfccs = librosa.feature.mfcc(y=audio, sr=target_sr, n_mfcc=40)
        chroma = librosa.feature.chroma_stft(y=audio, sr=target_sr)
        spec_contrast = librosa.feature.spectral_contrast(y=audio, sr=target_sr)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        chroma_scaled = np.mean(chroma.T, axis=0)
        spec_contrast_scaled = np.mean(spec_contrast.T, axis=0)
        combined_features = np.concatenate([mfccs_scaled, chroma_scaled, spec_contrast_scaled])
        features.append(combined_features)
        labels.append(row['category'])
    return np.array(features), np.array(labels)

data_path = "Combined"
metadata_path = "Complete.csv"

features, labels = load_data(data_path, metadata_path)
scaler = StandardScaler()
features = scaler.fit_transform(features)

le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
num_classes = len(le.classes_)
labels_onehot = np.eye(num_classes)[labels_encoded]

X_train, X_test, y_train, y_test = train_test_split(
    features, labels_onehot, test_size=0.21, random_state=37, stratify=labels_encoded
)

# Reshape for Conv1d: (batch, channels, length)
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

train_dataset = AudioDataset(X_train, y_train)
test_dataset = AudioDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# Model, loss, optimizer
model = ScreamNet(X_train.shape[2], num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

def train(model, loader, criterion, optimizer, epochs=30):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for x, y in loader:
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, torch.argmax(y, dim=1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(loader):.4f}")
    print("Training complete.")

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            outputs = model(x)
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == torch.argmax(y, dim=1)).sum().item()
    print(f"Test Accuracy: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    # Train and evaluate
    train(model, train_loader, criterion, optimizer, epochs=30)
    evaluate(model, test_loader)

    # Save the model for download
    os.makedirs('saved_models', exist_ok=True)
    torch.save(model.state_dict(), 'saved_models/screamnet.pth')
    print('Model saved as saved_models/screamnet.pth')
