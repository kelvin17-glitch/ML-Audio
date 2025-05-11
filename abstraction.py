import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

import librosa
import librosa.display
import IPython.display as ipd

from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from scipy.stats import skew, kurtosis

# Load audio file
class Visualizations:
    """
    Enables loading audio file, plotting Spectrograms and Mel Spectrograms

    Parameters
    ----------
    None
    """

    def plot_mag_spec(self, filepath, genre, n_fft=2048, hop_length=512):
        """
        Returns a Magnitude Spectrogram plot via librosa.display.specshow

        Parameters
        ----------
        filepath: str
                path to specific file eg, "./path/to/hiphop_001.wav"
        genre : str
                Name of Genre in question
        n_fft : int, default = 2048
                Number of FFT points. Also the window size
        hop_length : int, default = 512
                The number of samples between adjacent STFT columns
        """
        # Load file
        y, sr = librosa.load(filepath)
        
        # Perform STFT
        stft_result = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        
        # Calculate Spectogram
        mag_spec = np.abs(stft_result)
        
        # Convert Amplitude Spec to dB
        mag_db = librosa.amplitude_to_db(mag_spec, ref=np.max)
        
        # Magnitude Spectrogram Plot
        plt.figure(figsize=(15, 6))
        librosa.display.specshow(data=mag_db,
                                 sr=sr,
                                 hop_length=hop_length,
                                 x_axis="time",
                                 y_axis="log",
                                 cmap="viridis") # Changed from 'magma'
        plt.colorbar(format='%+2.0f dB')
        plt.title(f"{genre.title()} Magnitude Spectrogram");
            

    def plot_mel_spec(self, filepath, genre, n_fft=2048, hop_length=512, n_mels=128):
        """
        Returns a Mel Spectrogram plot

        Parameters
        ----------
        filepath: str
                path to specific file eg, "./path/to/hiphop_001.wav"
        genre : str
                Name of Genre in question
        n_fft : int, default = 2048
                Number of FFT points. Also the window size
        hop_length : int, default = 512
                The number of samples between adjacent STFT columns
        n_mels : int, default = 128
                Number of Mel bands (vertical resolution)
        """
        # Load file
        y, sr = librosa.load(filepath)
        
        # Initiate Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        
        # Scale to db
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Mel Spectrogram Plot
        plt.figure(figsize=(15, 6))
        librosa.display.specshow(data=mel_db,
                                 sr=sr,
                                 hop_length=hop_length,
                                 x_axis="time",
                                 y_axis="mel",
                                 cmap="viridis"),
        plt.colorbar(format='%+2.0f dB'),
        plt.title(f"{genre.title()} Mel Spectrogram");

    def plot_comparison_specs(self, filepath, genre, n_fft=2048, hop_length=512, n_mels=128):
        """
        Returns side-by-side Genre-specific Magnitude and Mel Spectogram plots

        Parameters
        ----------
        filepath: str
                path to specific file eg, "./path/to/hiphop_001.wav"
        genre : str
                Name of Genre in question
        n_fft : int, default = 2048
                Number of FFT points. Also the window size
        hop_length : int, default = 512
                The number of samples between adjacent STFT columns
        n_mels : int, default = 128
                Number of Mel for the Mel Spectrogram bands (vertical resolution)
        """
        # Load file
        y, sr = librosa.load(filepath)
        
        # Perform STFT
        stft_result = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        
        # Calculate Spectogram
        mag_spec = np.abs(stft_result)
        
        # Convert Amplitude Spec to dB
        mag_db = librosa.amplitude_to_db(mag_spec, ref=np.max)

        # Initiate Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        
        # Scale to db
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Plot Spectrogram
        fig, axs = plt.subplots(1, 2,figsize=(15, 6))
        img1 = librosa.display.specshow(data=mag_db,
                                 sr=sr,
                                 hop_length=hop_length,
                                 x_axis="time",
                                 y_axis="log",
                                 cmap="viridis",
                                 ax=axs[0])
        axs[0].set_title(f"{genre.title()} Magnitude Spectrogram")
        fig.colorbar(img1, ax=axs[0], format='%+2.0f dB');
        # Plot Mel Spectrogram
        img2 = librosa.display.specshow(data=mel_db,
                                 sr=sr,
                                 hop_length=hop_length,
                                 x_axis="time",
                                 y_axis="mel",
                                 cmap="viridis",
                                 ax=axs[1])
        axs[1].set_title(f"{genre.title()} Mel Spectrogram")
        fig.colorbar(img2, ax=axs[1], format='%+2.0f dB');

    # Plot Features functions
    # MEL-FREQUENCY CEPSTRAL COEFFICIENTS(MFCCs)
    def plot_mfccs(self, filepath, genre, n_mfcc=13, hop_length=512, n_fft=2048, n_mels=128):
        """
        Return a plot of the Mel-Frequency Cepstral Coefficients of audio file

        Parameters
        ----------
        filepath: str
                path to specific file eg, "./path/to/hiphop_001.wav"
        genre : str
                Name of Genre in question
        n_mfcc: int
                number of MFCCs you want
        hop_length : int, default = 512
                The number of samples between adjacent STFT columns
        n_fft : int, default = 2048
                Number of FFT points. Also the window size
        n_mels : int, default = 128
                Number of Mel for the Mel Spectrogram bands (vertical resolution)
        """
        # Load file
        y, sr = librosa.load(filepath)
        # Initialize MFCCs
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft, n_mels=n_mels)
        # Plot
        plt.figure(figsize=(15, 6))
        librosa.display.specshow(mfcc, x_axis="time", sr=sr)
        plt.title(f"{genre.title()} MFCCs")
        plt.colorbar().set_label("MFCC Magnitude")
        plt.ylabel("MFCC index")
        plt.tight_layout()
        plt.show()
        # Summary Stats
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

    # CHROMA FEATURES
    # Plot Chromagram
    def plot_chroma_stft(self, filepath, genre, hop_length=512):
        """
        Return a Chromagram plot of specific file
    
        Parameters
        ----------
        filepath: str
                path to specific file eg, "./path/to/hiphop_001.wav"
        genre : str
                 Name of Genre in question
        hop_length : int, default = 512
                The number of samples between adjacent STFT columns
        """
        # Load file
        y, sr = librosa.load(filepath)
        # Instatiate chroma_stft
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        # Plot
        plt.figure(figsize=(15,6))
        librosa.display.specshow(chroma_stft, y_axis="chroma", x_axis="time", hop_length=512)
        plt.title(f"{genre.title()} Chromagram")
        plt.colorbar()
        plt.tight_layout()

    # Plot Chroma Energy Normalized Statistics(chroma_cens)
    def plot_chroma_cens(self, filepath, genre, hop_length=512):
        """
        Return a Chroma Energy Normalized Statistics(chroma_cens) plot of specific file
    
        Parameters
        ----------
        filepath: str
                path to specific file eg, "./path/to/hiphop_001.wav"
        genre : str
                 Name of Genre in question
        hop_length : int, default = 512
                The number of samples between adjacent STFT columns
        """
       
        # Load file
        y, sr = librosa.load(filepath)
        
        # Instatiate chroma_cens
        chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
        
        # Plot
        plt.figure(figsize=(15,6))
        librosa.display.specshow(chroma_cens, y_axis="chroma", x_axis="time", hop_length=512)
        plt.title(f"{genre.title} Chroma CENS")
        plt.colorbar()
        plt.tight_layout()
        plt.show()

    # Plot the Chroma Constant-Q Transform feature(chroma_cqt)
    def plot_chroma_cqt(self, filepath, genre, hop_length=512):
        """
        Return a Chroma Constant-Q Transform(chroma_cqt) feature plot of specific file
    
        Parameters
        ----------
        filepath: str
                path to specific file eg, "./path/to/hiphop_001.wav"
        genre : str
                 Name of Genre in question
        hop_length : int, default = 512
                The number of samples between adjacent STFT columns
        """
        # Load file
        y, sr = librosa.load(filepath)
        
        # Instatiate chroma_cens
        chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
        
        # Plot
        plt.figure(figsize=(15,6))
        librosa.display.specshow(chroma_cqt, y_axis="chroma", x_axis="time", hop_length=512)
        plt.title(f"{genre.title} Chroma CQT")
        plt.colorbar()
        plt.tight_layout()
        plt.show()

class Features:
    """
    Extract and store features in a csv file

    Parameters
    ----------
    file_list : list 
            list of all files along with genre in a tuple e.g, [("../audio-ml/Data/genres_original/rock/rock.001.wav", rock)]
    """
    def __init__(self, file_list):
        self.file_list = file_list
        
    def __get_features(self, audio_path):
        """
        Helper function for feature extraction
        """
        # Load file
        y, sr = librosa.load(audio_path)
        
        # Initialize feature dict
        features = {}
        
        # Calculate RMS Energy and extract features
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        features["rms_mean"] = np.mean(rms)
        features["rms_std"] = np.std(rms)
        features["rms_max"] = np.max(rms)
        features["rms_min"] = np.min(rms)
        features["rms_skew"] = skew(rms)
        features["rms_kurtosis"] = kurtosis(rms)
        
        # Calculate Zero Crossing Rate and extract features
        zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=2048, hop_length=512)[0]
        features["zcr_mean"] = np.mean(zcr)
        features["zcr_std"] = np.std(zcr)
        features["zcr_max"] = np.max(zcr)
        features["zcr_min"] = np.min(zcr)
        features["zcr_skew"] = skew(zcr)
        features["zcr_kurtosis"] = kurtosis(zcr)  
        
        # Calculate Spectral Centroids and extract features
        centroids = librosa.feature.spectral_centroid(y=y, n_fft=2048, hop_length=512)[0]
        features["centroid_mean"] = np.mean(centroids)
        features["centroid_std"] = np.std(centroids)
        features["centroid_max"] = np.max(centroids)
        features["centroid_min"] = np.min(centroids)
        features["centroid_skew"] = skew(centroids)
        features["centroid_kurtosis"] = kurtosis(centroids)
        
        # Calculate Spectral Bandwidth and extract features
        bandwidth = librosa.feature.spectral_bandwidth(y=y, n_fft=2048, hop_length=512)[0]
        features["bandwidth_mean"] = np.mean(bandwidth)
        features["bandwidth_std"] = np.std(bandwidth)
        features["bandwidth_max"] = np.max(bandwidth)
        features["bandwidth_min"] = np.min(bandwidth)
        features["bandwidth_skew"] = skew(bandwidth)
        features["bandwidth_kurtosis"] = kurtosis(bandwidth) 
        
        # Calculate Spectral Contrast and extract features
        contrast = librosa.feature.spectral_contrast(y=y, n_fft=2048, hop_length=512)
        # For each band
        for i, band in enumerate(contrast):
            features[f"contrast{i}_mean"] = np.mean(band)
            features[f"contrast{i}_std"] = np.std(band)
            features[f"contrast{i}_max"] = np.max(band)
            features[f"contrast{i}_min"] = np.min(band)
            features[f"contrast{i}_skew"] = skew(band)
            features[f"contrast{i}_kurtosis"] = kurtosis(band)  
        
        # Calculate Spectral Rolloff and extract features
        rolloff = librosa.feature.spectral_rolloff(y=y, n_fft=2048, hop_length=512)[0]
        features["rolloff_mean"] = np.mean(rolloff)
        features["rolloff_std"] = np.std(rolloff)
        features["rolloff_max"] = np.max(rolloff)
        features["rolloff_min"] = np.min(rolloff)
        features["rolloff_skew"] = skew(rolloff)
        features["rolloff_kurtosis"] = kurtosis(rolloff)
    
        # Calculate MFCCs and extract features
        mfcc = librosa.feature.mfcc(y=y, sr=22050, n_mfcc=13, hop_length=512, n_fft=2048, n_mels=128)
        for i, coeff in enumerate(mfcc):
            features[f"mfcc{i+1}_mean"] = np.mean(coeff)
            features[f"mfcc{i+1}_std"] = np.std(coeff)
            features[f"mfcc{i+1}_min"] = np.min(coeff)
            features[f"mfcc{i+1}_max"] = np.max(coeff)
            features[f"mfcc{i+1}_skew"] = skew(coeff)
            features[f"mfcc{i+1}_kurtosis"] = kurtosis(coeff)
    
        # Calculate Chroma_cqt and extract features
        chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=22050)
        for i,v in enumerate(chroma_cqt):
            features[f"chroma_cqt{i}_mean"] = np.mean(v)
            features[f"chroma_cqt{i}_std"] = np.std(v)
            features[f"chroma_cqt{i}_min"] = np.min(v)
            features[f"chroma_cqt{i}_max"] = np.max(v)
            features[f"chroma_cqt{i}_skew"] = skew(v)
            features[f"chroma_cqt{i}_kurtosis"] = kurtosis(v)
        return features

    def extract_features(self, csv_name):
        """
        Extract features of audio files in file_list

        Parameters
        ----------
        csv_name: str
                custom name of csv file returned

        Returns
        ---------
        file: csv file containing features
        """
        # Extraction loop
        all_features = []
        all_labels = []
        
        for audio_path, label in tqdm(self.file_list):
            try:
                features = self.__get_features(audio_path) # Extract function
                # Add filename/label if needed
                features['filename'] = os.path.basename(audio_path)
                features['label'] = label
                all_features.append(features)
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
        
        feature_df = pd.DataFrame(all_features)
        
        # Save features in a csv file
        csv = feature_df.to_csv(csv_name, index=False)
        
        # Return the file
        return csv

class Model:
    """
    Train model, make predictions and evaluate accuracy

    Parameters
    ----------
    filepath: str
            path to csv file that contains extracted features e.g, "path/to/features.csv"
    """
    def __init__(self, filepath):
        self.filepath = filepath

    def make_predictions(self):
        """
        Train model, make and return predictions

        Parameters
        ----------
        None

        Returns
        ---------
        Array of predictions
        """
        # Load df
        df = pd.read_csv(self.filepath)
        
        # Split into feature matrix and target vector
        target = "label"
        X = df.drop(columns=["filename", "label"])
        y = df[target]
        
        # Perform a stratified train test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        self.y_test = y_test
        
        # Instantiate and fit scaler
        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Instantiate model
        model = SVC(kernel='rbf',gamma='scale', C=2.0)
        
        # Fit model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        self.y_pred = model.predict(X_test_scaled)

        # Return predictions
        return self.y_pred

    def evaluate_model(self, confusion_matrix=False):
        """
        Perform evaluation via three metrics: accuracy_score, classification_report and Confusion Matrix

        Parameters
        ----------
        confusion_matrix: bool, default=False
                Display the confusion matrix
        """
        # Final Model Evaluation
        print("--- Model Evaluation on Test Set ---")
        print("Final Accuracy:", accuracy_score(self.y_test, self.y_pred))
        print("\nFinal Classification Report:\n", classification_report(self.y_test, self.y_pred))
        
        if confusion_matrix:
            # Final model confusion matrix
            ConfusionMatrixDisplay.from_predictions(self.y_test, self.y_pred, xticks_rotation='vertical')
            plt.title('Final Model Confusion Matrix')
            plt.show()





        
    