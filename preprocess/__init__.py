import pandas as pd
import librosa
import librosa.display
import pathlib
import os
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
import json
import pandas as pd
import json
import shutil

from datetime import datetime



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
            print(f'Fold {i+1} processing Started')
            start_time = datetime.now()            
            files = os.listdir(f'{self.data_path}/fold{i+1}')
            pathlib.Path(path+'/data/temp/fold'+str(i+1)).mkdir(parents=True, exist_ok=True)
            test_count = 0
            start_time2 = datetime.now()
            for f in files:
                self.convert_to_mel(f'{self.data_path}/fold{i+1}/{f}',f'{path}/data/temp/fold{i+1}/{f[:-3]}png')
                test_count+=1
                if test_count%100==0:
                    end_time2 = datetime.now()
                    print(f'Processed {test_count} files')
                    print(f'Time take for processing is {end_time2 - start_time2}')
                    start_time2 = datetime.now()
    
            end_time = datetime.now()
            print(f'Fold {i+1} preparation is completed')
            print(f'No of files in the fold are {len(files)}')
            print(f'Time taken for preprocessing fold {i+1} {end_time - start_time}')
            print('')
            break

            
    def convert_to_mel(self,filepath,outpath):
        # print('Converting to mel started')
        # Loading the audio wave data using librosa
        data, sample_rate = librosa.load(filepath)
        # print(f'data shape = {data.shape} and sample rate = {sample_rate}')
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
        # print('Able to finish till the sepctogram')
        plt.axis('off')
        plt.savefig(outpath,bbox_inches='tight',  pad_inches=0)
        plt.clf()
        # transparent="True",

    def concurrent_prepare_audio_for_training(self):
        test_count = 1
        for i in range(10):
            print(f'Fold {i+1} processing Started')
            start_time = datetime.now()
            splits = list(self.create_splits(f'{self.data_path}/fold{i+1}').values())

            pathlib.Path(path+'/data/temp/fold'+str(i+1)).mkdir(parents=True, exist_ok=True)
            
            with concurrent.futures.ProcessPoolExecutor() as executor:
                executor.map(self.concurrent_processing, splits)
    
            end_time = datetime.now()
            print(f'Fold {i+1} preparation is completed')
            print(f'Time taken for preprocessing fold {i+1} {end_time - start_time}')
            print('')
            # break

    def create_splits(self,fold_path):
        fold_name = fold_path.split('/')[-1]
        json_path = '/'.join(fold_path.split('/')[:-1])
        files = os.listdir(fold_path)
        files = [f'{fold_path}/{f}' for f in files]
        splits =  np.array_split(files,8) # beacuse we have 8 cores
        out_json = {}
        for i in range(len(splits)):
            out_json[f'split_{i+1}'] = list(splits[i])

        with open(f'{json_path}/{fold_name}.json','w') as f:
            json.dump(out_json,f,indent=2)
        
        print(f'Spliting of {fold_name} is completed')

        return out_json
    
    def concurrent_processing(self,split):
        start_time = datetime.now()
        print(f'Start time is {start_time}')
        for f in split:
            output_filename = '/'.join(f.split('/')[-2:])[:-3] + 'png'
            self.convert_to_mel(f,f'{path}/data/temp/{output_filename}')
        
        end_time = datetime.now()
        print(f'End time is {start_time}')
        print(f'Time take for processing this split {end_time - start_time}')
    

    def create_training_data(self):
        print('Creating Final Data for ViT')
        df = pd.read_csv(f'{self.data_path}/UrbanSound8K.csv')
        df['file_name'] = df.apply(lambda x: x["slice_file_name"][:-3] + f'png' , axis = 1)
        df_final = df[['file_name','classID','class']]
        JSON_file = df_final.to_dict(orient='records')
        with open(path + '/data/train/metadata.jsonl', 'w') as outfile:
            for entry in JSON_file:
                json.dump(entry, outfile)
                outfile.write('\n')

        folders = os.listdir(f'{path}/data/temp')
        for folder in folders:
            self.copy_from_src_dst(f'{path}/data/temp/{folder}',f'{path}/data/train')

        id2label = df[['classID','class']].groupby('classID').agg({'class':'first'}).to_dict()['class']
        label2id = df[['classID','class']].groupby('class').agg({'classID':'first'}).to_dict()['classID']

        with open(path + '/data/id2label.json', 'w') as outfile:
            json.dump(id2label,outfile)
        
        with open(path + '/data/label2id.json', 'w') as outfile:
            json.dump(label2id,outfile)

        print('Completed creation')

    def copy_from_src_dst(self,src_dir,dest_dir):
        # getting all the files in the source directory
        
        os.system(f'cp -r {src_dir}/* {dest_dir}' )

    





# def process_image(img_name):
#     img = Image.open(img_name)

#     img = img.filter(ImageFilter.GaussianBlur(15))

#     img.thumbnail(size)
#     img.save(f'processed/{img_name}')
#     print(f'{img_name} was processed...')


# with concurrent.futures.ProcessPoolExecutor() as executor:
#     executor.map(process_image, img_names)