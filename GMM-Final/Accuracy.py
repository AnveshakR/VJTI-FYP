import os
import pandas as pd
import seaborn as sn
import _pickle as cPickle
from Predict import testPredict
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
 
modelpath = "Trained_Speech_Models/"

gmm_files = [os.path.join(modelpath,fname) for fname in 
              os.listdir(modelpath) if fname.endswith('.gmm')]

models    = [cPickle.load(open(fname,'rb')) for fname in gmm_files]
speakers   = [fname.split("/")[-1].split(".gmm")[0] for fname 
              in gmm_files]

print(speakers)

def getTrainFiles(train_speaker_folder):
    source   = "C:/Users/medee/Documents/COC/Speaker-Recognition-FYP/GMM-Final/Voice_Samples_Training/"   
    data_frame_row = []
    sub_folders = os.listdir(source)
    for folder in sub_folders:
        path_to_audio =  source + folder          
        data_frame_row.append([path_to_audio,train_speaker_folder])
    data_frame = pd.DataFrame(data_frame_row,columns=['audio_path', 'target_speaker'])
    return data_frame

def getTestFiles():
    source   = "C:/Users/medee/Documents/COC/Speaker-Recognition-FYP/GMM-Final/Testing_Audio/"   
    data_frame_row = []
    data_frame = pd.DataFrame()
    speaker_audio_folder = os.listdir(source)
    for folder in speaker_audio_folder:
        audio_files = os.listdir(source+folder)
        for file in audio_files:
            path_to_audio = source+folder+"/"+file
            data_frame_row.append([path_to_audio, speakers.index(folder)])
        data_frame = pd.DataFrame(data_frame_row,columns=['audio_path','actual'])
    return data_frame

def getActualPredictedList():
    data_frame_row = []
    testing_files =  getTestFiles()

    for index, row in testing_files.iterrows():
        audio_path = row["audio_path"]
        predicted = testPredict(audio_path)
        actual = row["actual"]
        data_frame_row.append([actual, predicted])
    actual_predicted = pd.DataFrame(data_frame_row,columns = ['actual','predicted']).sort_values(by='actual')
    return actual_predicted


def showAccuracyPlotAndMeasure():
    actual_pred = getActualPredictedList()
    actual = actual_pred["actual"].tolist()
    predicted = actual_pred["predicted"].tolist()
    labels  = sorted(actual_pred["actual"].unique().tolist())
    cm = confusion_matrix(y_true=actual, y_pred=predicted, labels=labels) 
    con_mat_df = pd.DataFrame(cm,index = speakers,columns = speakers)
    figure = plt.figure(figsize=(9,5))
    sn.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.title('Confusion matrix of GMM Model')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show()
    display_numeric_accuracy(actual, predicted, speakers)

def display_numeric_accuracy(actual,predicted,labels):
    print("\n")
    print(classification_report(y_true=actual, y_pred=predicted, target_names=labels))

showAccuracyPlotAndMeasure()