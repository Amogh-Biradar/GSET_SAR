# Scream_Detection.py
# Neural network for audio classification (scream detection)
# Uses MFCC, chroma, and spectral contrast features with a Conv1D-based model

import os
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv1D, MaxPooling1D, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
import wandb

# Initialize Weights & Biases for experiment tracking
wandb.init(
    project='audio_classification',
    name='gsetSAR',
    config={
        "batch_size": 64,  # Number of samples per gradient update
        "epochs": 120      # Number of training epochs
    },
    reinit=True  # Allows reinitialization if running multiple times
)

# Function to load and preprocess audio data
# Extracts MFCC, chroma, and spectral contrast features for each file
# Returns: features (numpy array), labels (numpy array)
def load_data(data_path, metadata_path):
    features = []
    labels = []
    metadata = pd.read_csv(metadata_path)
    for index, row in metadata.iterrows():
        file_path = row['filepath']
        print(file_path)  # Debug: print file being processed
        target_sr = 16000  # Target sample rate for all audio
        audio, sample_rate = librosa.load(file_path, sr=target_sr)
        # Feature extraction
        mfccs = librosa.feature.mfcc(y=audio, sr=target_sr, n_mfcc=40)
        chroma = librosa.feature.chroma_stft(y=audio, sr=target_sr)
        spec_contrast = librosa.feature.spectral_contrast(y=audio, sr=target_sr)
        # Take mean across time axis for each feature
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        chroma_scaled = np.mean(chroma.T, axis=0)
        spec_contrast_scaled = np.mean(spec_contrast.T, axis=0)
        # Concatenate all features into a single vector
        combined_features = np.concatenate([mfccs_scaled, chroma_scaled, spec_contrast_scaled])
        features.append(combined_features)
        labels.append(row['category'])
    return np.array(features), np.array(labels)

# Set data paths
data_path = "Combined"
metadata_path = "Complete.csv"

# Load and scale features
features, labels = load_data(data_path, metadata_path)
scaler = StandardScaler()
features = scaler.fit_transform(features)
print(f"[DEBUG] Loaded {len(features)} features and {len(labels)} labels")

# Check for empty dataset
if len(labels) == 0:
    raise ValueError("No data was loaded. Check paths and CSV file.")

# Encode string labels to integers, then to one-hot vectors
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
labels_onehot = to_categorical(labels_encoded)

# Split data into training and test sets (stratified by label)
X_train, X_test, y_train, y_test = train_test_split(
    features, labels_onehot, test_size=0.21, random_state=37, stratify=labels_encoded 
)

# Define model input shape
input_shape = (X_train.shape[1], 1)

# Build the neural network model
model = Sequential()
# First convolutional block
model.add(Conv1D(64, 3, padding='same', activation='relu', input_shape=input_shape))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.3))
# Second convolutional block
model.add(Conv1D(128, 3, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.3))
# Third convolutional block
model.add(Conv1D(256, 3, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.3))
# Flatten and dense layers
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))
# Output layer (softmax for multi-class classification)
model.add(Dense(len(le.classes_), activation='softmax'))

# Compile the model with Adam optimizer and categorical crossentropy loss
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0005), metrics=['accuracy'])

# Reshape data for Conv1D input
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Function to make predictions on a single audio file
# Returns the predicted class label

def make_predictions(model, le, file_path):
    audio, sample_rate = librosa.load(file_path, sr=16000)
    mfccs = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y=audio, sr=16000)
    spec_contrast = librosa.feature.spectral_contrast(y=audio, sr=16000)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    chroma_scaled = np.mean(chroma.T, axis=0)
    spec_contrast_scaled = np.mean(spec_contrast.T, axis=0)
    combined_features = np.concatenate([mfccs_scaled, chroma_scaled, spec_contrast_scaled])
    features = scaler.transform([combined_features])
    features = features.reshape(1, features.shape[1], 1)
    predicted_vector = model.predict(features)
    predicted_class_index = np.argmax(predicted_vector, axis=-1)
    return le.inverse_transform(predicted_class_index)[0]

# Set up W&B table for logging predictions
wandb_table = wandb.Table(columns=["File", "True Label", "Old Prediction", "New Prediction"])

# Save initial model weights for reference
initial_weights = model.get_weights()

# Store old predictions for comparison
old_predictions = {}

# List of test files and their true labels for evaluation
# (You can add more files as needed)
test_files = [
    ("Combined/zodTMCJFKv8_out.wav", "Screaming"),
    ("Combined/zxeeysNrEvM_out.wav", "NotScreaming"),
    ("Combined/zlt89JkjR8c_out.wav", "NotScreaming"),
    ("Combined/ySUuWxNEDtY_out.wav", "Screaming"),
]

# Make predictions before training (for comparison)
for file_path, true_label in test_files:
    predicted_label_before = make_predictions(model, le, file_path)
    old_predictions[file_path] = predicted_label_before

# Set up callbacks: early stopping and learning rate reduction
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

# Train the model
model.fit(
    X_train, y_train,
    batch_size=wandb.config.batch_size,
    epochs=wandb.config.epochs,
    validation_data=(X_test, y_test),
    verbose=1,
    callbacks=[wandb.keras.WandbCallback(), early_stop, reduce_lr]
)

# Make predictions after training and log to W
