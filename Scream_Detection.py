import os
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv1D, MaxPooling1D
import wandb

wandb.init(
    project='audio_classification',
    name='gsetSAR',
    config={
        "batch_size": 25,
        "epochs": 120
    },
    reinit=True  # allows reinitialization if you're in a notebook or looping
)

def load_data(data_path, metadata_path):
    features = []
    labels = []


    metadata = pd.read_csv(metadata_path)


    for index, row in metadata.iterrows():

        file_path = row['filepath']
        print(file_path)

        # Load the audio file and resample it
        target_sr = 44100
        audio, sample_rate = librosa.load(file_path, sr=target_sr)


        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=target_sr, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)


        # Append features and labels
        features.append(mfccs_scaled)
        labels.append(row['category'])


    return np.array(features), np.array(labels)

data_path = "Combined"
metadata_path = "Complete.csv"

features, labels = load_data(data_path, metadata_path)
print(f"[DEBUG] Loaded {len(features)} features and {len(labels)} labels")

if len(labels) == 0:
    raise ValueError("No data was loaded. Check paths and CSV file.")

# Encode labels
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
labels_onehot = to_categorical(labels_encoded)

X_train, X_test, y_train, y_test = train_test_split(
    features, labels_onehot, test_size=0.19, random_state=42, stratify=labels_encoded
)


input_shape = (X_train.shape[1], 1)
model = Sequential()
model.add(Conv1D(64, 3, padding='same', activation='relu', input_shape=input_shape))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.25))
model.add(Conv1D(128, 3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(le.classes_), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

def make_predictions(model, le, file_path):
    audio, sample_rate = librosa.load(file_path, sr=22050)
    mfccs = librosa.feature.mfcc(y=audio, sr=22050, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    features = mfccs_scaled.reshape(1, mfccs_scaled.shape[0], 1)
    predicted_vector = model.predict(features)
    predicted_class_index = np.argmax(predicted_vector, axis=-1)
    return le.inverse_transform(predicted_class_index)[0]

wandb_table = wandb.Table(columns=["File", "True Label", "Old Prediction", "New Prediction"])

initial_weights = model.get_weights()


# Dictionary to store old predictions
old_predictions = {}


# List of test files and their true labels
test_files = [
    ("Combined/zodTMCJFKv8_out.wav", "Screaming"),
    ("Combined/zxeeysNrEvM_out.wav", "NotScreaming"),
    ("Combined/zlt89JkjR8c_out.wav", "NotScreaming"),
    ("Combined/ySUuWxNEDtY_out.wav", "Screaming"),
    
]


# Make predictions before training
for file_path, true_label in test_files:
    predicted_label_before = make_predictions(model, le, file_path)
    old_predictions[file_path] = predicted_label_before

model.fit(X_train, y_train, batch_size=wandb.config.batch_size, epochs=wandb.config.epochs, 
          validation_data=(X_test, y_test), verbose=1, callbacks=[wandb.keras.WandbCallback()])

for file_path, true_label in test_files:
    predicted_label_after = make_predictions(model, le, file_path)
    wandb_table.add_data(file_path, true_label, old_predictions[file_path], predicted_label_after)




# Log the table to W&B
wandb.log({"Predictions": wandb_table})

wandb.finish()