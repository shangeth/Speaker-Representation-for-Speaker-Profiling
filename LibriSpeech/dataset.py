import os
import pandas as pd
import torch
import numpy as np
import random

import torchaudio
from torch.utils.data import Dataset
import wavencoder


# Unsupervised Dataset
class LibriDataset(Dataset):
    def __init__(self, hparams):
        self.root = os.path.join(hparams.un_data_path, 'final_repr_data')
        self.noise_dataset_path = hparams.noise_dataset_path
        self.wav_len = hparams.timit_wav_len

        self.speakers = os.listdir(self.root)
        self.speakersint = [int(x) for x in self.speakers]

        # Transforms
        self.train_transform = wavencoder.transforms.Compose([
            wavencoder.transforms.PadCrop(pad_crop_length=self.wav_len, pad_position='random', crop_position='random'),
            wavencoder.transforms.AdditiveNoise(self.noise_dataset_path, p=0.1),
            wavencoder.transforms.Clipping(p=0.1),
            ])

        self.info_file = os.path.join(hparams.un_data_path, 'SPEAKERS.TXT')
        df = pd.read_csv(self.info_file, skiprows=11, delimiter='|', error_bad_lines=False)
        df.columns = [col.strip().replace(';', '').lower() for col in df.columns]
        df = df.assign(
                sex=df['sex'].apply(lambda x: x.strip()),
                subset=df['subset'].apply(lambda x: x.strip()),
                name=df['name'].apply(lambda x: x.strip()),
                # id=df['id'].apply(lambda x: int(x))
            )
        self.info_df = df
        self.female_speakers = self.info_df.loc[self.info_df['subset']=='train-clean-360'].loc[self.info_df['sex'] == 'F']['id'].tolist()
        self.female_speakers = [int(x) for x in self.female_speakers]

        self.female_speakers = list(set(self.female_speakers) & set(self.speakersint))

    def __len__(self):
        return 124616 

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        

        # Anchor
        # if random.random() < 0.5:
        #     query_speaker = str(random.choice(self.female_speakers))
        # else:
        query_speaker = random.choice(self.speakers)
        filename = random.choice(os.listdir(os.path.join(self.root, query_speaker)))
        x, _ = torchaudio.load(os.path.join(self.root, query_speaker, filename))
        x = self.train_transform(x)

        # Positive 
        # if random.random()<0.1:
        #     xp, _ = torchaudio.load(os.path.join(self.root, query_speaker, filename))
        #     xp = self.train_transform(xp)
        # else:
        p_key_speaker = query_speaker
        filename = random.choice(os.listdir(os.path.join(self.root, p_key_speaker)))
        xp, _ = torchaudio.load(os.path.join(self.root, p_key_speaker, filename))
        xp = self.train_transform(xp)

        # Negative 
        n_key_speaker = random.choice(list(set(self.speakers) - set([query_speaker])))
        filename = random.choice(os.listdir(os.path.join(self.root, n_key_speaker)))
        xn, _ = torchaudio.load(os.path.join(self.root, n_key_speaker, filename))
        xn = self.train_transform(xn)
        
        return x, xp, xn