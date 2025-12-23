import os
import librosa
import numpy as np
import pickle

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# ---------------- CONFIG ----------------
AUDIO_DIR = "audio"
SAMPLE_RATE = 22050

# ---------------- FEATURE EXTRACTION ----------------
def extract_features(y, sr):
    feats = {}

    feats["tempo"] = librosa.beat.tempo(y=y, sr=sr)[0]
    feats["rms"] = np.mean(librosa.feature.rms(y=y))
    feats["zcr"] = np.mean(librosa.feature.zero_crossing_rate(y))
    feats["spectral_centroid"] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    feats["spectral_bandwidth"] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        feats[f"mfcc_{i+1}"] = np.mean(mfccs[i])

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    feats["chroma_mean"] = np.mean(chroma)

    return feats

# ---------------- LOAD DATA ----------------
X = []
y = []

labels = os.listdir(AUDIO_DIR)

for label in labels:
    label_path = os.path.join(AUDIO_DIR, label)
    if not os.path.isdir(label_path):
        continue

    for file in os.listdir(label_path):
        if file.endswith((".wav", ".mp3")):
            file_path = os.path.join(label_path, file)
            audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)

            feats = extract_features(audio, sr)
            X.append(list(feats.values()))
            y.append(label)

# ---------------- CONVERT ----------------
X = np.array(X)
y = np.array(y)

print("Samples:", X.shape[0])
print("Features:", X.shape[1])

# ---------------- ENCODE & SCALE ----------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# ---------------- TRAIN MODEL ----------------
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)
model.fit(X_scaled, y_encoded)

# ---------------- SAVE ARTIFACTS ----------------
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(encoder, open("label_encoder.pkl", "wb"))

# ✅ SAVE REAL FEATURE NAMES (CRITICAL FIX)
pickle.dump(
    list(feats.keys()),
    open("feature_names.pkl", "wb")
)

print("✅ Training complete. Model saved.")
