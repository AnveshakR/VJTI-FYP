import sounddevice as sd
from scipy.io.wavfile import write

def record():
    fs = 44100  # Sample rate
    seconds = 8  # Duration of recording
    source = 'Prediction_Audio/'
    filename = 'output.wav'
    path = source + filename
    print("Please speak......")
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()  # Wait until recording is finished
    write(path, fs, myrecording)  # Save as WAV file 
    print("output.wav succesfully saved in Testing_Audio folder")
    return filename