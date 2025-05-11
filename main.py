# Import libraries
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

from glob import glob

import librosa
import librosa.display

from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from abstraction import Features, Model


# Paths to variables
blues_files = glob("../audio-ml/Data/genres_original/blues/blues.*.wav")

classical_files = glob("../audio-ml/Data/genres_original/classical/classical.*.wav")

country_files = glob("../audio-ml/Data/genres_original/country/country.*.wav")

disco_files = glob("../audio-ml/Data/genres_original/disco/disco.*.wav")

hiphop_files = glob("../audio-ml/Data/genres_original/hiphop/hiphop.*.wav")

jazz_files = glob("../audio-ml/Data/genres_original/jazz/jazz.*.wav")

metal_files = glob("../audio-ml/Data/genres_original/metal/metal.*.wav")

pop_files = glob("../audio-ml/Data/genres_original/pop/pop.*.wav")

reggae_files = glob("../audio-ml/Data/genres_original/reggae/reggae.*.wav")

rock_files = glob("../audio-ml/Data/genres_original/rock/rock.*.wav")

# Create a (path, genre) list of all audio files
file_list = []
# Iteration
for path in classical_files:
    file_list.append((path, "classical"))
for path in blues_files:
    file_list.append((path, "blues"))
for path in country_files:
    file_list.append((path, "country"))
for path in disco_files:
    file_list.append((path, "disco"))
for path in hiphop_files:
    file_list.append((path, "hiphop"))
for path in jazz_files:
    file_list.append((path, "jazz"))
for path in metal_files:
    file_list.append((path, "metal"))
for path in pop_files:
    file_list.append((path, "pop"))
for path in reggae_files:
    file_list.append((path, "reggae"))
for path in rock_files:
    file_list.append((path, "rock"))

# Instantiate features class
feature = Features(file_list=file_list)
# Feature Extraction
feature.extract_features("audio_features.csv")

# Instantiate Model class
model = Model("./audio_features.csv")
# Make predictions
predictions = model.make_predictions()

# Evaluate model
model.evaluate_model()