import os
import librosa
import numpy as np
import pandas as pd
import librosa.display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def extract_features(file_path, sr=22050):
    try:
        y, sr = librosa.load(file_path, sr=sr)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return np.mean(mfccs, axis=1)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def load_data(data_path):
    genres = os.listdir(data_path)
    features, labels = [], []
    
    for genre in genres:
        genre_path = os.path.join(data_path, genre)
        if not os.path.isdir(genre_path):
            continue
        
        for file in os.listdir(genre_path):
            file_path = os.path.join(genre_path, file)
            feature = extract_features(file_path)
            if feature is not None:
                features.append(feature)
                labels.append(genre)
    
    return np.array(features), np.array(labels)

def train_model(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Model Accuracy:", accuracy)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    joblib.dump(model, "genre_classifier.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(le, "label_encoder.pkl")
    print("Model and preprocessing objects saved!")

def predict_genre(file_path):
    model = joblib.load("genre_classifier.pkl")
    scaler = joblib.load("scaler.pkl")
    le = joblib.load("label_encoder.pkl")
    
    feature = extract_features(file_path)
    if feature is None:
        return "Error extracting features"
    
    feature_scaled = scaler.transform([feature])
    genre_index = model.predict(feature_scaled)[0]
    return le.inverse_transform([genre_index])[0]

if __name__ == "__main__":
    data_path = "./genres"  # Path to dataset containing genre folders
    X, y = load_data(data_path)
    train_model(X, y)
    
    test_file = input("Enter the path of a song to predict its genre: ")
    print("Predicted Genre:", predict_genre(test_file))