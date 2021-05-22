import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

import pytorch_lightning as pl
from pytorch_lightning.metrics.regression import MeanAbsoluteError as MAE
from pytorch_lightning.metrics.regression import MeanSquaredError  as MSE

from TIMIT.dataset import TIMITDataset
from LibriSpeech.dataset import LibriDataset
from Model.model import Encoder, Discriminator, Accumulator, ProfilerH
from Model.utils import RMSELoss

from config import TIMITConfig
import os
import pandas as pd


class LightningModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        HPARAMS = vars(hparams)
        self.HPARAMS = HPARAMS

        self.E = Encoder()
        self.A = Accumulator(HPARAMS['hidden_size'])
        self.profiler = ProfilerH(HPARAMS['hidden_size'])
        self.D = Discriminator(2*HPARAMS['hidden_size'])

        self.classification_criterion = MSE()
        self.regression_criterion = MSE()
        self.mae_criterion = MAE()
        self.rmse_criterion = RMSELoss()

        self.lr = HPARAMS['lr']

        self.csv_path = HPARAMS['speaker_csv_path']
        self.df = pd.read_csv(self.csv_path)
        self.h_mean = self.df['height'].mean()
        self.h_std = self.df['height'].std()

        # Dataloader for unsupervised dataset
        libri_train_set = LibriDataset(hparams=hparams)
        self.libriloader = data.DataLoader(
                    libri_train_set, 
                    batch_size=64, 
                    shuffle=True, 
                    num_workers=4)

    def forward(self, x):
        return self.E(x)

    def configure_optimizers(self):
        # params = [
            # {'params': self.E.parameters(), 'lr': 1e-4}, 
            # {'params': list(self.D.parameters()) + list(self.A.parameters()) + list(self.profiler.parameters()), 'lr': 1e-3}
            # ]
        optimizer = optim.Adam(self.parameters())
        return [optimizer]

    def training_step(self, batch, batch_idx):
        timit_batch = batch 
        libri_batch = next(iter(self.libriloader))

        x, y_h, y_a, y_g = timit_batch
        xl, xpl, xnl = libri_batch
        xl, xpl, xnl = xl.to(self.device), xpl.to(self.device), xnl.to(self.device)

        # Unsupervised Representation learning Path
        z = self(xl)
        zp = self(xpl)
        zn = self(xnl)

        z_a = self.A(z)
        zp_a = self.A(zp)
        zn_a = self.A(zn)

        yp = self.D(z_a, zp_a)
        yn = self.D(z_a, zn_a)

        loss_p = self.classification_criterion(yp, torch.ones_like(yp, device=self.device)) 
        loss_n = self.classification_criterion(yn, torch.zeros_like(yn, device=self.device))
        js_loss = loss_p + loss_n
        repr_loss = js_loss 

        
        # Consistency Path
        y_hat_hx = self.profiler(z_a)
        y_hat_hxp = self.profiler(zp_a)
        ss_loss = self.regression_criterion(y_hat_hx, y_hat_hxp)

        # Supervised speaker profiling Path
        z_profiling = self(x)
        z_profiling = self.A(z_profiling)
        y_hat_h = self.profiler(z_profiling)
        y_h, y_a, y_g = y_h.view(-1).float(), y_a.view(-1).float(), y_g.view(-1).float()
        y_hat_h = y_hat_h.view(-1).float()

        height_loss = self.regression_criterion(y_hat_h, y_h)
        profiling_loss = height_loss 

        height_mae = self.mae_criterion(y_hat_h*self.h_std+self.h_mean, y_h*self.h_std+self.h_mean)

        loss = repr_loss + profiling_loss + ss_loss

        return {'loss':loss, 
                'repr_loss' : repr_loss,
                'profiling_loss' : profiling_loss,
                'train_height_mae':height_mae.item(),
                'consistency_loss':ss_loss
                }
    
    def training_epoch_end(self, outputs):
        n_batch = len(outputs)
        loss = torch.tensor([x['loss'] for x in outputs]).mean()
        repr_loss = torch.tensor([x['repr_loss'] for x in outputs]).mean()
        profiling_loss = torch.tensor([x['profiling_loss'] for x in outputs]).mean()
        consistency_loss = torch.tensor([x['consistency_loss'] for x in outputs]).mean()

        height_mae = torch.tensor([x['train_height_mae'] for x in outputs]).sum()/n_batch

        self.log('train/loss' , loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train/repr' , repr_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train/profiling' , profiling_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train/consistency' , consistency_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train/h',height_mae.item(), on_step=False, on_epoch=True, prog_bar=False)

    def validation_step(self, batch, batch_idx):
        x, y_h, y_a, y_g = batch
        z = self.E(x)
        z = self.A(z)
        y_hat_h = self.profiler(z)
        y_h, y_a, y_g = y_h.view(-1).float(), y_a.view(-1).float(), y_g.view(-1).float()
        y_hat_h = y_hat_h.view(-1).float()

        height_loss = self.regression_criterion(y_hat_h, y_h)
        loss = height_loss

        height_mae = self.mae_criterion(y_hat_h*self.h_std+self.h_mean, y_h*self.h_std+self.h_mean)


        return {'val_loss':loss, 
                'val_height_mae':height_mae.item()}

    def validation_epoch_end(self, outputs):
        n_batch = len(outputs)
        val_loss = torch.tensor([x['val_loss'] for x in outputs]).mean()
        height_mae = torch.tensor([x['val_height_mae'] for x in outputs]).sum()/n_batch
        
        self.log('val/loss' , val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/h',height_mae.item(), on_step=False, on_epoch=True, prog_bar=True)


