import os
from tabnanny import check
import _pickle as cPickle
import numpy as np
from scipy.io.wavfile import read
from FeatureExtraction import extract_features
from Predict import testPredict
from Record_Audio import record
#from speakerfeatures import extract_features
import warnings
warnings.filterwarnings("ignore")
import time


#path to training data
source   = "Build_Set/"   
modelpath = "Testing_Models/"
test_file = "Build_Set_Text.txt"        
file_paths = open(test_file,'r')


#path to training data
source   = "Testing_Audio/"   

prediction_source = "Prediction_Audio/"

#path where training speakers will be saved
modelpath = "Trained_Speech_Models/"

gmm_files = [os.path.join(modelpath,fname) for fname in 
              os.listdir(modelpath) if fname.endswith('.gmm')]
print(gmm_files)
 
#Load the Gaussian gender Models
models    = [cPickle.load(open(fname,'rb')) for fname in gmm_files]
speakers   = [fname.split("/")[-1].split(".gmm")[0] for fname 
              in gmm_files]
print(speakers)

error = 0
total_sample = 0.0

print("Press '0' for testing a complete set of audio with Accuracy or Press '1' for checking a single Audio or Press '2' for testing live audio (microphone required):")
take=int(input().strip())
if take == 2:
    filename = record()
    print (("Testing Audio : ",filename))
    sr,audio = read(prediction_source + filename)
    vector   = extract_features(audio,sr)
    
    log_likelihood = np.zeros(len(models)) 
    
    for i in range(len(models)):
        gmm    = models[i]  #checking with each model one by one
        scores = np.array(gmm.score(vector))
        log_likelihood[i] = scores.sum()
    
    winner = np.argmax(log_likelihood)
    print ("\tThe live audio sample of person is detected as - ", speakers[winner])
elif take == 1:
    print ("Enter the File name from the sample with .wav notation :")
    path =input().strip()
    winner = testPredict(path, prediction_source)
    print ("\tThe person in the given audio sample is detected as - ", speakers[winner])
    time.sleep(1.0)
elif take == 0:
    test_file = "Testing_audio_Path.txt"        
    file_paths = open(test_file,'r')
    # Read the test directory and get the list of test audio files 
    for path in file_paths:   
        total_sample+= 1.0
        path=path.strip()
        print("Testing Audio : ", path)
        sr,audio = read(source + path)
        vector   = extract_features(audio,sr)
        log_likelihood = np.zeros(len(models)) 
        for i in range(len(models)):
            gmm    = models[i]  #checking with each model one by one
            scores = np.array(gmm.score(vector))
            log_likelihood[i] = scores.sum()
        winner=np.argmax(log_likelihood)
        print ("\tdetected as - ", speakers[winner])
        checker_name = path.split("/")[0]
        if speakers[winner] != checker_name:
            error += 1
        time.sleep(1.0)
    print ("error-"+str(error), "\ttotal-sample-"+str(total_sample))
    accuracy = ((total_sample - error) / total_sample) * 100
    print ("The Accuracy Percentage for the current testing Performance with MFCC + GMM is : ", accuracy, "%")


print ("Speaker Identified Successfully")
