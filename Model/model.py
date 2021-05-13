import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wavencoder
from transformers import Wav2Vec2Model

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = wavencoder.models.Wav2Vec(pretrained=True)
        # Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").feature_extractor

    def forward(self, x):
        x = self.feature_extractor(x.squeeze(1)) # [Batch, D, N]
        x = x.transpose(1,2) # [Batch, N, D]
        return x


class Accumulator(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.dim = dim
        self.lstm = nn.LSTM(512, self.dim, batch_first=True)
        # self.attention = wavencoder.layers.SoftAttention(self.dim, self.dim)

    def forward(self, x):
        output, (hidden, _) = self.lstm(x) # [Batch, N, D]
        # attn_output = self.attention(output) # [Batch, D]
        attn_output = output[:, -1, :] # [Batch, D]
        return attn_output

class Discriminator(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.dim = dim
        # self.bn = nn.BatchNorm1d(num_features=self.dim)
        self.classifier = nn.Sequential(
            nn.Linear(int(self.dim),  int(self.dim/4)),
            nn.ReLU(),
            nn.Linear(int(self.dim/4), 1),
            nn.Sigmoid()
        )

    def forward(self, z1, z2): 
        z = torch.cat([z1, z2], 1)
        # z = self.bn(z)
        y_hat = self.classifier(z)
        return y_hat


class Profiler(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        lstm_h = dim

        self.height_regressor = nn.Sequential(
            nn.Linear(lstm_h,  int(lstm_h/4)),
            nn.ReLU(),
            nn.Linear(int(lstm_h/4), 1),
        )
        self.age_regressor = nn.Sequential(
            nn.Linear(lstm_h,  int(lstm_h/4)),
            nn.ReLU(),
            nn.Linear(int(lstm_h/4), 1),
            # nn.Linear(lstm_h,  1),
        )
        self.gender_classifier = nn.Sequential(
            nn.Linear(lstm_h,  int(lstm_h/4)),
            nn.ReLU(),
            nn.Linear(int(lstm_h/4), 1),
            # nn.Linear(lstm_h,  1),
        )

    def forward(self, z):
        height = self.height_regressor(z)
        age = self.age_regressor(z)
        gender = self.gender_classifier(z)
        return height, age, gender


class ProfilerH(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        lstm_h = dim

        self.height_regressor = nn.Sequential(
            nn.Linear(lstm_h,  int(lstm_h/4)),
            nn.ReLU(),
            nn.Linear(int(lstm_h/4), 1),
        )

    def forward(self, z):
        height = self.height_regressor(z)
        return height