import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wavencoder
from transformers import Wav2Vec2Model

class SEncoder(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        inp_dim = 40
        self.feature_extractor = nn.Sequential(
            nn.BatchNorm1d(inp_dim),
            nn.Conv1d(inp_dim, dim, 3),
            nn.ReLU(),
            nn.BatchNorm1d(dim),
            nn.MaxPool1d(2,2),
            nn.Conv1d(dim, dim, 3),
            nn.ReLU(),
            nn.BatchNorm1d(dim),
            nn.MaxPool1d(2,2),
            nn.Conv1d(dim, dim, 3),
            nn.ReLU(),
            nn.BatchNorm1d(dim),
            nn.MaxPool1d(2,2), 
        )

    def forward(self, x):
        x = self.feature_extractor(x.squeeze(1)) # [Batch, D, N]
        x = x.transpose(1,2) # [Batch, N, D]
        return x


class SAccumulator(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.dim = dim
        self.lstm = nn.LSTM(self.dim, self.dim, batch_first=True)
        # self.attention = wavencoder.layers.SoftAttention(self.dim, self.dim)

    def forward(self, x):
        output, (hidden, _) = self.lstm(x) # [Batch, N, D]
        # attn_output = self.attention(output) # [Batch, D]
        attn_output = output[:, -1, :] # [Batch, D]
        return attn_output