from time import sleep
import tkinter
from tkinter import filedialog
from numpy import rint
import sounddevice as sd
import soundfile as sf
from tkinter import *
import os
from pydub import AudioSegment
from pathlib import PurePath
from playsound import playsound
import tensorflow as tf 
import librosa
import numpy as np
import warnings
warnings.filterwarnings("ignore")

audio_path = 'temp_test.wav'
model = tf.keras.models.load_model("final-model")

spkr_id = {
    1:'Dixit',
    2:'Sourav',
    3:'Anveshak',
    4:'Vedant',
    5:'Darshan'
}

def voice_rec():
    
    global audio_path
    audio_path = "temp_test.wav"
    selected_sample['text'] = audio_path

    fs = 48000
    duration = 8
    print("Speak now....")
    myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=2)
    sd.wait()
    
    if os.path.exists('temp_test.wav'):
        os.remove('temp_test.wav')
    if os.path.exists('temp_test.flac'):
        os.remove('temp_test.flac')

    sf.write('temp_test.flac', myrecording, fs)
    path = PurePath(r'temp_test.flac')
    data = AudioSegment.from_file(path,path.suffix[1:])
    data.export(path.name.replace(path.suffix, "") + ".wav", format="wav")
    os.remove('temp_test.flac')

    return None

def select_file():

    global audio_path
    select_tk = Tk()
    file_extensions = ['*.wav', '*.WAV', '*.Wav']
    ftypes = [('WAV files', file_extensions), ('All files', "*")]
    select_tk.filename = filedialog.askopenfilename(title="Select audio file", filetypes=ftypes)
    audio_path = select_tk.filename
    selected_sample.config(text = audio_path)
    select_tk.destroy()

    return None
  
def play_sound():

    print(audio_path)
    playsound(audio_path)

# INTERFACE WITH OUTSIDE MODEL HERE
def get_prediction():

   
    load_audio,_ = librosa.load(audio_path,sr=16000)
    load_audio = load_audio[:112000]
    load_audio = [load_audio.tolist()]
    load_audio_arr = np.array(load_audio)
    pred_label = model.predict(load_audio_arr)

    pred = np.argmax(pred_label)

    return spkr_id[pred]

def insert_pred():

    Pred.config(text = get_prediction())

master = Tk()

master.title("FYP")

Label(master, text="Choose Sample: ").grid(row=0, sticky=W, rowspan=2)
record_btn = Button(master, text="Record", command = voice_rec)
record_btn.grid(row=0, column=2)
Label(master, text="or").grid(row=0, column=3)
select_btn = Button(master, text="Select", command = select_file)
select_btn.grid(row=0, column=4)

Label(master, text="Selected Sample:").grid(row=3)
selected_sample = Label(master, text=' ')
selected_sample.grid(row=3, column = 2)

Label(master, text="Check Sample: ").grid(row = 4,sticky = W, rowspan=2)
play_btn = Button(master, text="Listen", command = play_sound)
play_btn.grid(row=4, column=2, columnspan=2, rowspan=2, padx=5, pady=5)

Pred = Label(master)
Pred.grid(row=6, column=2, columnspan=2, rowspan=2, padx=5, pady=5)
Button(master, text="Get Prediction: ", command = insert_pred).grid(row = 6,column = 0, sticky = W, rowspan=2)

mainloop()