'''
Generate prediction files for resnet models in the following format:

Argument specifies destination directory in which this script
generates content with following contents:

- targets.npy
- 1.npy
- 2.npy
 .
 .
 .
- 10.npy

where each seed prediction file contains a numpy array: [num_samples x num_classes]
'''

import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from scipy import stats
from typing import Dict
from train_resnet import normalize, get_lab_to_ind, ResNet, get_default_device

@torch.no_grad()
def get_predictions(dl, model, device):
    '''
    Returns predictions for specified model
    Classification: [num_samples x num_classes]
    '''
    model.eval()
    preds_list = []
    for (x,_) in dl:
        x = x.to(device)
        logits = model(x)
        s = nn.Softmax(dim=1)
        probs = s(logits)
        preds_list.append(probs)
    preds = torch.cat(preds_list).detach().cpu().numpy()
    return preds

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='resnet predict.')
    parser.add_argument('trained_models_dir', type=str, help='Path to dir of trained models')
    parser.add_argument('out_dir', type=str, help='Path to dir to save prediction files')
    parser.add_argument('data_paths', type=str, help='list of all eval data files')
    parser.add_argument('train_data_path', type=str, help='train data file')

    args = parser.parse_args()
    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/resnet_predict.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    data_paths = args.data_paths.split()
    dfs = []
    for data_path in data_paths:
        dfs.append(pd.read_csv(data_path))
    df = pd.concat(dfs)

    # Need train data for normalizing constants
    df_train = pd.read_csv(args.train_data_path)

    # Identify the categorical features
    cat_features = []
    for col in df:
        values = df[col].tolist()
        unique = list(dict.fromkeys(values))
        if len(unique) < 20:
            cat_features.append(col)

    nan_replacements = {}
    for col in df_train:
        if col in cat_features:
            nan_replacements[col] = stats.mode(np.asarray(df_train[col].tolist()))[0][0]
        else:
            nan_replacements[col] = np.mean(np.asarray(df_train[col].dropna().tolist()))

    # Replace nans in train and dev
    for col in df_train:
        df_train[col] = df_train[col].fillna(nan_replacements[col])
        df[col] = df[col].fillna(nan_replacements[col])

    print("Loaded Data")

    # 10 models assumed
    ensemble_size = 10

    # Get the device
    device = get_default_device()

    # Generate prediction file for each model
    for seed in range(1, ensemble_size+1):
        # Set Seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
    
        model = ResNet()
        model.load_state_dict(torch.load(f'{args.trained_models_dir}/seed{seed}.th', map_location='cpu'))
        model.to(device)

        # Normalize data using specific seed

        # Normalise using train data stats
        # Quantile normalisation is used (maps to a normal distribution)
        X_train_np = np.asarray(df_train.iloc[:,6:])
        X_dev_np = np.asarray(df.iloc[:,6:])
        X = {'train': X_train_np, 'dev': X_dev_np}
        X = normalize(X, normalization='quantile', seed=seed)
        X_train_np = X['train']
        X_dev_np = X['dev']

        X_train = torch.FloatTensor(X_train_np)
        X_dev = torch.FloatTensor(X_dev_np)

        lab_to_ind = get_lab_to_ind(df_train)
        batch_size = 512

        # get targets for dev
        y_dev = df['fact_cwsm_class']
        y_dev = torch.LongTensor(np.asarray([lab_to_ind[lab] for lab in y_dev]))
        labels = y_dev.detach().cpu().numpy()
        np.save(f'{args.out_dir}/targets.npy', labels)

        dev_ds = TensorDataset(X_dev, y_dev)
        dev_dl = DataLoader(dev_ds, batch_size=batch_size, shuffle=False)

        # get prediction
        preds = np.asarray(get_predictions(dev_dl, model, device))
        np.save(f'{args.out_dir}/{seed}.npy', preds)

        print('Done seed', seed)

