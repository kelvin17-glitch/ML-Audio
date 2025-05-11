# 🎵 Track Genre Classification using Machine Learning

This project uses the GTZAN dataset and a Support Vector Machine (SVM) to classify music tracks into 10 genres based on audio features such as MFCCs and chroma features.

## 🚀 Features
- Audio feature extraction with `librosa`
- Preprocessing and normalization
- Multi-class classification using SVM
- Achieved **~82.5% accuracy**

## 🎧 Genres Covered
- Blues, Classical, Country, Disco, Hip-Hop, Jazz, Metal, Pop, Reggae, Rock

## 🛠️ Tech Stack
- Python
- `librosa` for audio processing
- `scikit-learn` for modeling
- `matplotlib` / `seaborn` for visualizations

## 🧠 How It Works
1. Load and preprocess GTZAN dataset
2. Extract MFCCs and chroma features
3. Train/test split
4. Train an SVM classifier
5. Evaluate performance

## 📈 Results
- Accuracy: **82.5%**
- Confusion matrix shows strong performance on most genres, with some confusion between similar ones (e.g., Rock vs. Metal)

## 📂 Structure
project-root/
│
├── data/ # GTZAN dataset
├── notebooks/ # Jupyter notebooks
├── main.py / # Main script
├── report.txt/ # Report, figures, etc.
├── abstraction.py # Py Classes for visualization, model training & prediction, and feature extraction
├── README.md # You're here
└─ requirements.txt # Dependencies


## 🧪 How to Run
```bash
pip install -r requirements.txt
python main.py
