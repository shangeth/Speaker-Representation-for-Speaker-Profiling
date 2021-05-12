class TIMITConfig(object):
    # path to the unzuipped TIMIT data folder
    data_path = '/home/shangeth/DATASET/TIMIT/wav_data'

    # path to csv file containing age, heights of timit speakers
    speaker_csv_path = '/home/shangeth/Speakerprofiling/src/Dataset/data_info_height_age.csv'

    # unsupervised libri data path
    un_data_path = '/home/shangeth/DATASET/LibriSpeech/LibriSpeech'

    # length of wav files for training and testing
    timit_wav_len = 16000 * 1

    batch_size = 128
    epochs = 200
    
    # loss = alpha * height_loss + beta * age_loss + gamma * gender_loss
    alpha = 1
    beta = 1
    gamma = 1

    # data type - raw/spectral
    data_type = 'raw'

    # model type
    ## AHG 
    # wav2vecLSTMAttn/spectralCNNLSTM/MultiScale

    model_type = 'MultiScale'

    # AHG or only H
    training_type = 'AHG'

    # hidden dimension of LSTM and Dense Layers
    hidden_size = 256

    # No of GPUs for training and no of workers for datalaoders
    gpu = '-1'
    n_workers = 4

    # model checkpoint to continue from or test on
    model_checkpoint = None
    
    # noise dataset for augmentation
    noise_dataset_path = '/home/shangeth/noise_dataset'

    # LR of optimizer
    lr = 1e-3

    run_name = data_type + '_' + training_type + '_' + model_type 