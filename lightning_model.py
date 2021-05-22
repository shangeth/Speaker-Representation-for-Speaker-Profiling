import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

import pytorch_lightning as pl
from pytorch_lightning.metrics.regression import MeanAbsoluteError as MAE
from pytorch_lightning.metrics.regression import MeanSquaredError  as MSE
from pytorch_lightning.metrics.classification import Accuracy

from TIMIT.dataset import TIMITDataset
from LibriSpeech.dataset import LibriDataset
from Model.model import Encoder, Discriminator, Accumulator, Profiler
from Model.spectral_model import SEncoder, SAccumulator
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

        if HPARAMS['model_type'] == 'raw':
            self.E = Encoder(HPARAMS['hidden_size'])
            self.A = Accumulator(HPARAMS['hidden_size'])
            self.profiler = Profiler(HPARAMS['hidden_size'])
            self.D = Discriminator(2*HPARAMS['hidden_size'])
        else:
            self.E = SEncoder(HPARAMS['hidden_size'])
            self.A = SAccumulator(HPARAMS['hidden_size'])
            self.profiler = Profiler(HPARAMS['hidden_size'])
            self.D = Discriminator(2*HPARAMS['hidden_size'])


        self.classification_criterion = MSE()
        self.regression_criterion = MSE()
        self.mae_criterion = MAE()
        self.rmse_criterion = RMSELoss()
        self.accuracy = Accuracy()

        self.alpha = HPARAMS['alpha']
        self.beta = HPARAMS['beta']
        self.gamma = HPARAMS['gamma']

        self.lr = HPARAMS['lr']

        self.csv_path = HPARAMS['speaker_csv_path']
        self.df = pd.read_csv(self.csv_path)
        self.h_mean = self.df['height'].mean()
        self.h_std = self.df['height'].std()
        self.a_mean = self.df['age'].mean()
        self.a_std = self.df['age'].std()

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
        y_hat_hx, y_hat_ax, y_hat_gx = self.profiler(z_a)
        y_hat_hxp, y_hat_axp, y_hat_gxp = self.profiler(zp_a)
        ss_loss = self.regression_criterion(y_hat_hx, y_hat_hxp)  +  self.regression_criterion(y_hat_ax, y_hat_axp) + self.regression_criterion(y_hat_gx, y_hat_gxp)

        # Supervised speaker profiling Path
        z_profiling = self(x)
        z_profiling = self.A(z_profiling)
        y_hat_h, y_hat_a, y_hat_g = self.profiler(z_profiling)
        y_h, y_a, y_g = y_h.view(-1).float(), y_a.view(-1).float(), y_g.view(-1).float()
        y_hat_h, y_hat_a, y_hat_g = y_hat_h.view(-1).float(), y_hat_a.view(-1).float(), y_hat_g.view(-1).float()

        height_loss = self.regression_criterion(y_hat_h, y_h)
        age_loss = self.regression_criterion(y_hat_a, y_a)
        gender_loss = self.classification_criterion(y_hat_g, y_g)
        profiling_loss = self.alpha * height_loss + self.beta * age_loss + self.gamma * gender_loss

        height_mae = self.mae_criterion(y_hat_h*self.h_std+self.h_mean, y_h*self.h_std+self.h_mean)
        age_mae =self.mae_criterion(y_hat_a*self.a_std+self.a_mean, y_a*self.a_std+self.a_mean)
        gender_acc = self.accuracy((y_hat_g>0.5).long(), y_g.long())

        loss = repr_loss + profiling_loss + ss_loss
        # if self.current_epoch < 100:
        #     loss = 100 * repr_loss + profiling_loss + ss_loss
        # else:
        #     loss = repr_loss + profiling_loss + 100 * ss_loss

        #schedule1
        # if self.current_epoch < 100:
        #     loss = 100*repr_loss + profiling_loss + ss_loss
        # else:
        #     loss = repr_loss + profiling_loss + 100 * ss_loss


        return {'loss':loss, 
                'repr_loss' : repr_loss,
                'profiling_loss' : profiling_loss,
                'train_height_mae':height_mae.item(),
                'train_age_mae':age_mae.item(),
                'train_gender_acc':gender_acc,
                'consistency_loss':ss_loss
                }
    
    def training_epoch_end(self, outputs):
        n_batch = len(outputs)
        loss = torch.tensor([x['loss'] for x in outputs]).mean()
        repr_loss = torch.tensor([x['repr_loss'] for x in outputs]).mean()
        profiling_loss = torch.tensor([x['profiling_loss'] for x in outputs]).mean()
        consistency_loss = torch.tensor([x['consistency_loss'] for x in outputs]).mean()

        height_mae = torch.tensor([x['train_height_mae'] for x in outputs]).sum()/n_batch
        age_mae = torch.tensor([x['train_age_mae'] for x in outputs]).sum()/n_batch
        gender_acc = torch.tensor([x['train_gender_acc'] for x in outputs]).mean()

        self.log('train/loss' , loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train/repr' , repr_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train/profiling' , profiling_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train/consistency' , consistency_loss, on_step=False, on_epoch=True, prog_bar=False)

        self.log('train/h',height_mae.item(), on_step=False, on_epoch=True, prog_bar=False)
        self.log('train/a',age_mae.item(), on_step=False, on_epoch=True, prog_bar=False)
        self.log('train/g',gender_acc, on_step=False, on_epoch=True, prog_bar=False)

    def validation_step(self, batch, batch_idx):
        x, y_h, y_a, y_g = batch
        z = self.E(x)
        z = self.A(z)
        y_hat_h, y_hat_a, y_hat_g = self.profiler(z)
        y_h, y_a, y_g = y_h.view(-1).float(), y_a.view(-1).float(), y_g.view(-1).float()
        y_hat_h, y_hat_a, y_hat_g = y_hat_h.view(-1).float(), y_hat_a.view(-1).float(), y_hat_g.view(-1).float()

        height_loss = self.regression_criterion(y_hat_h, y_h)
        age_loss = self.regression_criterion(y_hat_a, y_a)
        gender_loss = self.classification_criterion(y_hat_g, y_g)
        loss = self.alpha * height_loss + self.beta * age_loss + self.gamma * gender_loss

        height_mae = self.mae_criterion(y_hat_h*self.h_std+self.h_mean, y_h*self.h_std+self.h_mean)
        age_mae = self.mae_criterion(y_hat_a*self.a_std+self.a_mean, y_a*self.a_std+self.a_mean)
        gender_acc = self.accuracy((y_hat_g>0.5).long(), y_g.long())

        return {'val_loss':loss, 
                'val_height_mae':height_mae.item(),
                'val_age_mae':age_mae.item(),
                'val_gender_acc':gender_acc}

    def validation_epoch_end(self, outputs):
        n_batch = len(outputs)
        val_loss = torch.tensor([x['val_loss'] for x in outputs]).mean()
        height_mae = torch.tensor([x['val_height_mae'] for x in outputs]).sum()/n_batch
        age_mae = torch.tensor([x['val_age_mae'] for x in outputs]).sum()/n_batch
        gender_acc = torch.tensor([x['val_gender_acc'] for x in outputs]).mean()
        
        self.log('val/loss' , val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/h',height_mae.item(), on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/a',age_mae.item(), on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/g',gender_acc, on_step=False, on_epoch=True, prog_bar=True)

        # if self.current_epoch > 10:
        #     for param in self.E.parameters():
        #         param.requires_grad = True

    # def test_step(self, batch, batch_idx):
    #     x, y_h, y_a, y_g = batch
    #     z = self.E(x)
    #     z = self.A(z)
    #     y_hat_h, y_hat_a, y_hat_g = self.profiler(z)
    #     y_h, y_a, y_g = y_h.view(-1).float(), y_a.view(-1).float(), y_g.view(-1).float()
    #     y_hat_h, y_hat_a, y_hat_g = y_hat_h.view(-1).float(), y_hat_a.view(-1).float(), y_hat_g.view(-1).float()

    #     gender_acc = self.accuracy((y_hat_g>0.5).long(), y_g.long())

    #     idx = y_g.view(-1).long()
    #     female_idx = torch.nonzero(idx).view(-1)
    #     male_idx = torch.nonzero(1-idx).view(-1)

    #     male_height_mae = self.mae_criterion(y_hat_h[male_idx]*self.h_std+self.h_mean, y_h[male_idx]*self.h_std+self.h_mean)
    #     male_age_mae = self.mae_criterion(y_hat_a[male_idx]*self.a_std+self.a_mean, y_a[male_idx]*self.a_std+self.a_mean)

    #     femal_height_mae = self.mae_criterion(y_hat_h[female_idx]*self.h_std+self.h_mean, y_h[female_idx]*self.h_std+self.h_mean)
    #     female_age_mae = self.mae_criterion(y_hat_a[female_idx]*self.a_std+self.a_mean, y_a[female_idx]*self.a_std+self.a_mean)

    #     male_height_rmse = self.rmse_criterion(y_hat_h[male_idx]*self.h_std+self.h_mean, y_h[male_idx]*self.h_std+self.h_mean)
    #     male_age_rmse = self.rmse_criterion(y_hat_a[male_idx]*self.a_std+self.a_mean, y_a[male_idx]*self.a_std+self.a_mean)

    #     femal_height_rmse = self.rmse_criterion(y_hat_h[female_idx]*self.h_std+self.h_mean, y_h[female_idx]*self.h_std+self.h_mean)
    #     female_age_rmse = self.rmse_criterion(y_hat_a[female_idx]*self.a_std+self.a_mean, y_a[female_idx]*self.a_std+self.a_mean)

    #     return {
    #             'male_height_mae':male_height_mae.item(),
    #             'male_age_mae':male_age_mae.item(),
    #             'female_height_mae':femal_height_mae.item(),
    #             'female_age_mae':female_age_mae.item(),
    #             'male_height_rmse':male_height_rmse.item(),
    #             'male_age_rmse':male_age_rmse.item(),
    #             'femal_height_rmse':femal_height_rmse.item(),
    #             'female_age_rmse':female_age_rmse.item(),
    #             'test_gender_acc':gender_acc}

    # def test_epoch_end(self, outputs):
    #     n_batch = len(outputs)
    #     male_height_mae = torch.tensor([x['male_height_mae'] for x in outputs]).mean()
    #     male_age_mae = torch.tensor([x['male_age_mae'] for x in outputs]).mean()
    #     female_height_mae = torch.tensor([x['female_height_mae'] for x in outputs]).mean()
    #     female_age_mae = torch.tensor([x['female_age_mae'] for x in outputs]).mean()

    #     male_height_rmse = torch.tensor([x['male_height_rmse'] for x in outputs]).mean()
    #     male_age_rmse = torch.tensor([x['male_age_rmse'] for x in outputs]).mean()
    #     femal_height_rmse = torch.tensor([x['femal_height_rmse'] for x in outputs]).mean()
    #     female_age_rmse = torch.tensor([x['female_age_rmse'] for x in outputs]).mean()

    #     gender_acc = torch.tensor([x['test_gender_acc'] for x in outputs]).mean()

    #     pbar = {'male_height_mae' : male_height_mae.item(),
    #             'male_age_mae':male_age_mae.item(),
    #             'female_height_mae':female_height_mae.item(),
    #             'female_age_mae': female_age_mae.item(),
    #             'male_height_rmse' : male_height_rmse.item(),
    #             'male_age_rmse':male_age_rmse.item(),
    #             'femal_height_rmse':femal_height_rmse.item(),
    #             'female_age_rmse': female_age_rmse.item(),
    #             'test_gender_acc':gender_acc.item()}
    #     self.logger.log_hyperparams(pbar)
    #     self.log_dict(pbar)

