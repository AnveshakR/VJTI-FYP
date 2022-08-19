import os
import numpy as np
from scipy.io.wavfile import read
from FeatureExtraction import extract_features
import matplotlib.pyplot as plt

data_dir = 'C:/Users/medee/Documents/COC/Speaker-Recognition-FYP/GMM-Final/Voice_Samples_Training/'
all_dir = os.listdir(data_dir)

filenameArray = []
spkrFeatures = []
spkrName = ['Anveshak','Darshan','Dixit','Sourav','Vedant']

for folder in all_dir:
    for file in os.listdir(data_dir + folder):
        filenameArray.append(file)
        break

for i in range(len(filenameArray)):
    path = data_dir + all_dir[i] + "/" + filenameArray[i].strip()   
    sr,audio = read(path)
    vector   = extract_features(audio,sr)
    vector = np.array(vector).T
    # print("Vector: ",vector[1])
    print("Vector.shape: ",vector.shape)
    spkrFeatures.append(vector[1])

X = np.arange(0, 801, 1)
plt.plot(X, spkrFeatures[0][:801], color='r', label=spkrName[0], linewidth=0.5)
plt.xlabel("X----->")
plt.ylabel("Y----->")
plt.title("First cepstrum coefficient plot for "+spkrName[0]+"'s audio")
plt.show()

plt.plot(X, spkrFeatures[1][:801], color='b', label=spkrName[1], linewidth=0.5)
plt.xlabel("X----->")
plt.ylabel("Y----->")
plt.title("First cepstrum coefficient plot for "+spkrName[1]+"'s audio")
plt.show()

plt.plot(X, spkrFeatures[2][:801], color='g', label=spkrName[2], linewidth=0.5)
plt.xlabel("X----->")
plt.ylabel("Y----->")
plt.title("First cepstrum coefficient plot for "+spkrName[2]+"'s audio")
plt.show()

plt.plot(X, spkrFeatures[3][:801], color='c', label=spkrName[3], linewidth=0.5)
plt.xlabel("X----->")
plt.ylabel("Y----->")
plt.title("First cepstrum coefficient plot for "+spkrName[3]+"'s audio")
plt.show()

plt.plot(X, spkrFeatures[4][:801], color='y', label=spkrName[4], linewidth=0.5)
plt.xlabel("X----->")
plt.ylabel("Y----->")
plt.title("First cepstrum coefficient plot for "+spkrName[4]+"'s audio")
plt.show()