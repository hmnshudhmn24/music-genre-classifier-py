# Music Genre Classifier

This project predicts the genre of a song using machine learning techniques. It extracts audio features using **Librosa**, processes them with **Scikit-Learn**, and classifies them with a **Random Forest model**.

## Features
- Extracts **MFCC features** from audio files.
- Supports multiple **music genres**.
- Trains a **Random Forest Classifier**.
- Saves and loads **trained models**.
- Predicts the genre of a given audio file.

## Requirements
Install dependencies before running the project:
```sh
pip install numpy pandas librosa scikit-learn joblib matplotlib
```

## Dataset Structure
Place the dataset in a folder named `genres/`, where each subfolder is a genre containing audio files.
```
genres/
  ├── rock/
  │   ├── song1.wav
  │   ├── song2.wav
  │   └── ...
  ├── jazz/
  │   ├── song1.wav
  │   ├── song2.wav
  │   └── ...
  └── ...
```

## Usage
1. **Train the model**:
```sh
python music_genre_classifier.py
```

2. **Predict a song's genre**:
```sh
python music_genre_classifier.py
```
When prompted, enter the path of the song to classify.

## Model Output
- Saves the trained model as `genre_classifier.pkl`
- Saves the **Scaler** and **Label Encoder** for preprocessing future predictions.

## Notes
- Ensure all audio files are in a format readable by Librosa (`.wav` recommended).
- The accuracy of predictions depends on the quality and diversity of the dataset.

## License
This project is open-source and free to use for learning and research purposes.