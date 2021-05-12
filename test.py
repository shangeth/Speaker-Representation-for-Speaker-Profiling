from config import TIMITConfig

from argparse import ArgumentParser
from multiprocessing import Pool
import os

from TIMIT.dataset import TIMITDataset
from lightning_model import LightningModel

from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
import pytorch_lightning as pl

import torch
import torch.utils.data as data

from tqdm import tqdm 
import pandas as pd
import numpy as np


if __name__ == "__main__":

    parser = ArgumentParser(add_help=True)
    parser.add_argument('--data_path', type=str, default=TIMITConfig.data_path)
    parser.add_argument('--speaker_csv_path', type=str, default=TIMITConfig.speaker_csv_path)
    parser.add_argument('--un_data_path', type=str, default=TIMITConfig.un_data_path)
    parser.add_argument('--timit_wav_len', type=int, default=TIMITConfig.timit_wav_len)
    parser.add_argument('--batch_size', type=int, default=TIMITConfig.batch_size)
    parser.add_argument('--epochs', type=int, default=TIMITConfig.epochs)
    parser.add_argument('--alpha', type=float, default=TIMITConfig.alpha)
    parser.add_argument('--beta', type=float, default=TIMITConfig.beta)
    parser.add_argument('--gamma', type=float, default=TIMITConfig.gamma)
    parser.add_argument('--hidden_size', type=float, default=TIMITConfig.hidden_size)
    parser.add_argument('--lr', type=float, default=TIMITConfig.lr)
    parser.add_argument('--gpu', type=int, default=TIMITConfig.gpu)
    parser.add_argument('--n_workers', type=int, default=TIMITConfig.n_workers)
    parser.add_argument('--dev', type=str, default=False)
    parser.add_argument('--model_checkpoint', type=str, default=TIMITConfig.model_checkpoint)
    parser.add_argument('--noise_dataset_path', type=str, default=TIMITConfig.noise_dataset_path)
    parser.add_argument('--model_type', type=str, default=TIMITConfig.model_type)
    parser.add_argument('--training_type', type=str, default=TIMITConfig.training_type)
    parser.add_argument('--data_type', type=str, default=TIMITConfig.data_type)


    parser = pl.Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()
    print(f'Testing Model on NISP Dataset\n#Cores = {hparams.n_workers}\t#GPU = {hparams.gpu}')

    # Testing Dataset
    test_set = TIMITDataset(
        wav_folder = os.path.join(hparams.data_path, 'TEST'),
        hparams = hparams,
        is_train=False
    )

    csv_path = hparams.speaker_csv_path
    df = pd.read_csv(csv_path)
    h_mean = df['height'].mean()
    h_std = df['height'].std()
    a_mean = df['age'].mean()
    a_std = df['age'].std()

    #Testing the Model
    if hparams.model_checkpoint:
        if TIMITConfig.training_type == 'AHG':
            model = LightningModel.load_from_checkpoint(hparams.model_checkpoint, hparams=hparams)
            model.eval()
            height_pred = []
            height_true = []
            age_pred = []
            age_true = []
            gender_pred = []
            gender_true = []


            # i = 0 
            for batch in tqdm(test_set):
                x, y_h, y_a, y_g = batch
                # y_hat_h, y_hat_a, y_hat_g = model(x)
                y_hat_h, y_hat_a, y_hat_g = model.profiler(model.A(model(x)))

                height_pred.append((y_hat_h*h_std+h_mean).item())
                age_pred.append((y_hat_a*a_std+a_mean).item())
                gender_pred.append(y_hat_g>0.5)

                height_true.append((y_h*h_std+h_mean).item())
                age_true.append(( y_a*a_std+a_mean).item())
                gender_true.append(y_g)

                # if i> 5: break
                # i += 1
            female_idx = np.where(np.array(gender_true) == 1)[0].reshape(-1).tolist()
            male_idx = np.where(np.array(gender_true) == 0)[0].reshape(-1).tolist()

            height_true = np.array(height_true)
            height_pred = np.array(height_pred)
            age_true = np.array(age_true)
            age_pred = np.array(age_pred)


            hmae = mean_absolute_error(height_true[male_idx], height_pred[male_idx])
            hrmse = mean_squared_error(height_true[male_idx], height_pred[male_idx], squared=False)
            amae = mean_absolute_error(age_true[male_idx], age_pred[male_idx])
            armse = mean_squared_error(age_true[male_idx], age_pred[male_idx], squared=False)
            print(hrmse, hmae, armse, amae)

            hmae = mean_absolute_error(height_true[female_idx], height_pred[female_idx])
            hrmse = mean_squared_error(height_true[female_idx], height_pred[female_idx], squared=False)
            amae = mean_absolute_error(age_true[female_idx], age_pred[female_idx])
            armse = mean_squared_error(age_true[female_idx], age_pred[female_idx], squared=False)
            print(hrmse, hmae, armse, amae)

            print(accuracy_score(gender_true, gender_pred))
        
        else:
            model = LightningModel.load_from_checkpoint(hparams.model_checkpoint, HPARAMS=vars(hparams))
            model.eval()
            height_pred = []
            height_true = []
            gender_true = []

            for batch in tqdm(test_set):
                x, y_h, y_a, y_g = batch
                # y_hat_h = model(x)
                y_hat_h = model.profiler(model.A(model(x)))

                height_pred.append((y_hat_h*h_std+h_mean).item())
                height_true.append((y_h*h_std+h_mean).item())
                gender_true.append(y_g)

            female_idx = np.where(np.array(gender_true) == 1)[0].reshape(-1).tolist()
            male_idx = np.where(np.array(gender_true) == 0)[0].reshape(-1).tolist()

            height_true = np.array(height_true)
            height_pred = np.array(height_pred)

            hmae = mean_absolute_error(height_true[male_idx], height_pred[male_idx])
            hrmse = mean_squared_error(height_true[male_idx], height_pred[male_idx], squared=False)
            print(hrmse, hmae)

            hmae = mean_absolute_error(height_true[female_idx], height_pred[female_idx])
            hrmse = mean_squared_error(height_true[female_idx], height_pred[female_idx], squared=False)
            print(hrmse, hmae)


    else:
        print('Model chekpoint not found for Testing !!!')