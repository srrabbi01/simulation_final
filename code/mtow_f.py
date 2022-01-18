import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from pydub import AudioSegment
import glob

def convert_audio(audio_path, target_path):
    os.system(f"ffmpeg -i {audio_path} -ac 1 -ar 16000 {target_path}")

def m4atowav():
    path_in_mp3 = './raw_data/' #path to folder mp3 files
    path_out_16 = './raw_data/process_data/' #path to folder wav files (converted to wav and downsample, samplerate = outrate)
    for file in os.listdir(path_in_mp3):
        print(file)
        if file.endswith(".m4a"):
            sound = AudioSegment.from_file(path_in_mp3 + file) #load mp3 file
            file48 = path_in_mp3+file.replace('.m4a','.wav')
            sound.export(file48, format="wav")
            convert_audio(file48,path_out_16 + file.replace('.m4a','.wav'))
            if os.path.isfile(file48):
                os.remove(file48)
# def mp3towav():
#     path_in_mp3 = './raw_data/' #path to folder mp3 files
#     path_out_16 = './raw_data/process_data/' #path to folder wav files (converted to wav and downsample, samplerate = outrate)
#     for file in os.listdir(path_in_mp3):
#         print(file)
#         if file.endswith(".aac"):
#             sound = AudioSegment.from_file(path_in_mp3 + file) #load mp3 file
#             file48 = path_in_mp3+file.replace('.aac','.wav')
#             sound.export(file48, format="wav")
#             convert_audio(file48,path_out_16 + file.replace('.aac','.wav'))
#             if os.path.isfile(file48):
#                 os.remove(file48)
def mp3towav():
    path_in_mp3 = './raw_data/' #path to folder mp3 files
    path_out_16 = './raw_data/process_data/' #path to folder wav files (converted to wav and downsample, samplerate = outrate)
    for file in os.listdir(path_in_mp3):
        print(file)
        if file.endswith(".mp3"):
            sound = AudioSegment.from_mp3(path_in_mp3 + file) #load mp3 file
            file48 = path_in_mp3+file.replace('.mp3','.wav')
            sound.export(file48, format="wav")
            convert_audio(file48,path_out_16 + file.replace('.mp3','.wav'))
            if os.path.isfile(file48):
                os.remove(file48)
        

def data_process():
    processpath = "./raw_data/process_data/"
    filterpath = "./filter_data/"
    for file in os.listdir(processpath):
        print(file)
        if file.endswith(".wav"):

            song = AudioSegment.from_wav(processpath+file)
            new = song.low_pass_filter(1000)
            new1 = new.high_pass_filter(1000)
            song_6_db_quieter = new1 - 15
            extrafile = processpath+"file.wav"
            song_6_db_quieter.export(extrafile, "wav")

            offset = 10000
            data, samplerate = sf.read(extrafile)
            fft = np.fft.fft(data)
            # Arange the data given and extract a midpoint
            x = np.arange(0, samplerate, samplerate/len(fft))
            mid = len(x)/2
            # Set variables to know when to stop offset to left and right
            i = int(mid - offset)  
            # Offset all values starting at left and stopping once we reach right offset
            while(i < (mid + offset)):
                 fft[i] = 0
                 i += 1
             # Apply inverse of FFt to cleaned signal
            clean = np.fft.ifft(fft)
            # Get only real part of signal for new sound file
            clean = np.real(clean)
            sf.write(filterpath+file, clean, samplerate)
            if os.path.isfile(extrafile):
                os.remove(extrafile)


mp3towav()
data_process()