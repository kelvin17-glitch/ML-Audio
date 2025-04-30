import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from glob import glob

import librosa
import librosa.display
import IPython.display as ipd

# Load audio file
class Input:
    """
    Enables loading audio file, plotting Spectrograms and Mel Spectrograms

    Parameters
    ----------
    filepath : str 
            genre[idx] e.g, hiphop[1]
    sr : int, default = 22050
            the sampling rate
    """
    def __init__(self, filepath, sr=22050):
        self.filepath = filepath
        self.sr = sr

    def load_file(self, n_fft=2048, hop_length=512):
        """
        Loads file, returns Magnitude Spec index as an instance attribute

        Parameters
        ----------
        n_fft : int, default = 2048
                Number of FFT points. Also the window size
        hop_length : int, default = 512
                The number of samples between adjacent STFT columns
        """
        # Load file
        y, sr = librosa.load(self.filepath, sr=None)
        self.y = y
        # Perform STFT
        stft_result = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        # Calculate Spectogram
        mag_spec = np.abs(stft_result)
        # Convert Amplitude & Power Specs to dB
        self.mag_db = librosa.amplitude_to_db(mag_spec, ref=np.max)

    def plot_mag_spec(self, genre, hop_length=512):
        """
        Returns a Magnitude Spectrogram plot via librosa.display.specshow

        Parameters
        ----------
        genre : str
                Name of Genre in question
        hop_length : int, default = 512
                The number of samples between adjacent STFT columns
        """
        # Magnitude Spectrogram Plot
        plt.figure(figsize=(15, 6))
        librosa.display.specshow(data=self.mag_db,
                                 sr=self.sr,
                                 hop_length=hop_length,
                                 x_axis="time",
                                 y_axis="log",
                                 cmap="viridis") # Changed from 'magma'
        plt.colorbar(format='%+2.0f dB')
        plt.title(f"{genre.title()} Magnitude Spectrogram");
            

    def plot_mel_spec(self, genre, n_fft=2048, hop_length=512, n_mels=128):
        """
        Returns a Mel Spectrogram plot

        Parameters
        ----------
        genre : str
                Name of Genre in question
        n_fft : int, default = 2048
                Number of FFT points. Also the window size
        hop_length : int, default = 512
                The number of samples between adjacent STFT columns
        n_mels : int, default = 128
                Number of Mel bands (vertical resolution)
        """
        # Initiate Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(y=self.y, sr=self.sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        # Scale to db
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        # Mel Spectrogram Plot
        plt.figure(figsize=(15, 6))
        librosa.display.specshow(data=mel_db,
                                 sr=self.sr,
                                 hop_length=hop_length,
                                 x_axis="time",
                                 y_axis="mel",
                                 cmap="viridis"),
        plt.colorbar(format='%+2.0f dB'),
        plt.title(f"{genre.title()} Mel Spectrogram");