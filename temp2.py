import pandas as pd
import json

df = pd.read_csv('/home/ubuntu/audio_transformer/data/original/urbansound8k/UrbanSound8K.csv')

df['file_name'] = df.apply(lambda x: x["slice_file_name"][:-3] + f'png' , axis = 1)

df_fold1 = df[df['fold']==1][['file_name','classID','class']]
json_file = df_fold1.to_dict(orient='records')

import json
with open('/home/ubuntu/audio_transformer/data/train/metadata.json') as f:
    JSON_file = json.load(f)

with open('/home/ubuntu/audio_transformer/data/train/metadata.jsonl', 'w') as outfile:
    for entry in JSON_file:
        json.dump(entry, outfile)
        outfile.write('\n')

def create_json(p1,p2):
    
    