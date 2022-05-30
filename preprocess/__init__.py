import pandas as pd
import librosa
import librosa.display
import pathlib
import os
import numpy as np
import matplotlib.pyplot as plt

path = str(pathlib.Path(__file__).parents[1])
pathlib.Path(path+'/data').mkdir(parents=True, exist_ok=True)
pathlib.Path(path+'/data/train').mkdir(parents=True, exist_ok=True)
pathlib.Path(path+'/data/test').mkdir(parents=True, exist_ok=True)
pathlib.Path(path+'/data/original').mkdir(parents=True, exist_ok=True)
pathlib.Path(path+'/data/temp').mkdir(parents=True, exist_ok=True)

class Preprocess:
    def __init__(self):
        self.is_data_downloaded = False
        self.get_data_from_web()
        self.data_path = path+'/data/original/urbansound8k'
        
    def get_data_from_web(self):
        if pathlib.Path(path+'/data/original/urbansound8k').exists():
            self.is_data_downloaded = True
            return True
        else:
            self.is_data_downloaded = False
            print(f'Data is not yet downloaded, follow README to download the data.')
            print('')
            return False

    def prepare_audio_for_training(self):
        test_count = 1
        for i in range(10):
            files = os.listdir(f'{self.data_path}/fold{i+1}')
            pathlib.Path(path+'/data/temp/fold'+str(i+1)).mkdir(parents=True, exist_ok=True)
            for f in files:
                self.convert_to_mel(f'{self.data_path}/fold{i+1}/{f}',f'{path}/data/temp/fold{i+1}/{f[:-3]}png')
                test_count+=1
                if test_count>=1:
                    break
            break

            print(f'Fold {i+1} preparation is completed')
    
    def convert_to_mel(self,filepath,outpath):
        print('Converting to mel started')
        # Loading the audio wave data using librosa
        data, sample_rate = librosa.load(filepath)
        print(f'data shape = {data.shape} and sample rate = {sample_rate}')
        
        # Computing the spectogram
        sgram = librosa.stft(data)

        # use the mel-scale instead of raw frequency
        sgram_mag, _ = librosa.magphase(sgram)
        mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=sample_rate)

        # use the decibel scale to get the final Mel Spectrogram
        mel_sgram = librosa.amplitude_to_db(mel_scale_sgram, ref=np.min)
        # librosa.display.specshow(mel_sgram, sr=sample_rate, x_axis='time', y_axis='mel')
        # plt.colorbar(format='%+2.0f dB')
        # print(mel_sgram)
        _ = librosa.display.specshow(mel_sgram, sr=sample_rate)
        print('Able to finish till the sepctogram')
        plt.savefig(outpath,bbox_inches='tight', transparent="True", pad_inches=0)
