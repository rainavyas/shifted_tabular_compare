'''
Evaluate Single and Ensemble
Wrt accuracy and macro-f1 score
'''

import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from evaluation_tools import get_accuracy, get_avg_f1
from train_mlp import MLP
from scipy import stats
from typing import Dict
import sklearn.preprocessing
from train_mlp import normalize, get_lab_to_ind

@torch.no_grad()
def get_predictions(dl, model):
    '''
    Returns predictions for specified model
    Classification: [num_samples x num_classes]
    '''
    model.eval()
    preds_list = []
    for (x,_) in dl:
        logits = model(x)
        s = nn.Softmax(dim=1)
        probs = s(logits)
        preds_list.append(probs)
    preds = torch.cat(preds_list).detach().cpu().numpy()
    return preds

if __name__ == '__main__':

    DATA = 'dev'
    # DATA = 'test'

    # Load the trained models
    dir_path = './trained_mlp_classification_models'
    models = []

    # Load the data
    df_in = pd.read_csv(f'../data/{DATA}_in.csv')
    df_out = pd.read_csv(f'../data/{DATA}_out.csv')
    #df = pd.concat([df_in, df_out])
    df = pd.concat([df_in])
    # df = pd.concat([df_out])

    # Need train data for normalizing constants
    df_train = pd.read_csv('../data/train.csv')

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




    # 10 models provided
    ensemble_size = 10

    # collect all the predictions from each model
    all_preds = []
    all_labels = []


    for seed in range(1, ensemble_size+1):

        # Set Seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        model = MLP()
        model.load_state_dict(torch.load(f'{dir_path}/seed{seed}.th', map_location='cpu'))
        models.append(model)

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
        batch_size = 1024

        # # Train
        # y_train = np.asarray(df_train['fact_cwsm_class'])
        # y_train = torch.LongTensor(np.asarray([lab_to_ind[lab] for lab in y_train]))

        # train_ds = TensorDataset(X_train, y_train)
        # train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        # Dev
        y_dev = df['fact_cwsm_class']
        y_dev = torch.LongTensor(np.asarray([lab_to_ind[lab] for lab in y_dev]))
        labels = y_dev.detach().cpu().numpy()
        all_labels.append(labels)

        dev_ds = TensorDataset(X_dev, y_dev)
        dev_dl = DataLoader(dev_ds, batch_size=batch_size, shuffle=False)

        # get prediction
        preds = np.asarray(get_predictions(dev_dl, model))
        all_preds.append(preds)

    all_preds = np.stack(all_preds, axis=0)
    all_labels = np.stack(all_labels, axis=0)

    # Accuracy
     
    # Ensemble
    ens_preds = np.squeeze(np.mean(all_preds, axis=0))
    acc_ens = get_accuracy(ens_preds, labels)

    # Single
    sing_accs = []
    for i in range(ensemble_size):
        curr_preds = all_preds[i]
        acc = get_accuracy(curr_preds, labels)
        sing_accs.append(acc)
    
    sing_accs = np.asarray(sing_accs)
    acc_sing_avg = np.mean(sing_accs)
    acc_sing_std = np.std(sing_accs)

    # Report accuracies
    print("--------------")
    print("Accuracy")
    print(f"Ensemble: {acc_ens}")
    print(f"Single: {acc_sing_avg} +- {acc_sing_std}")
    print("-------------")

    # F1-vs all average across all classes

    # Ensemble
    f1_ens = get_avg_f1(ens_preds, labels)

    # Single
    sing_f1s = []
    for i in range(ensemble_size):
        curr_preds = all_preds[i]
        f1 = get_avg_f1(curr_preds, labels)
        sing_f1s.append(f1)
    
    sing_f1s = np.asarray(sing_f1s)
    f1_sing_avg = np.mean(sing_f1s)
    f1_sing_std = np.std(sing_f1s)

    # Report F1s
    print("--------------")
    print("F1")
    print(f"Ensemble: {f1_ens}")
    print(f"Single: {f1_sing_avg} +- {f1_sing_std}")
    print("-------------")


        
