import argparse
import os
import sys
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict
import sklearn.preprocessing
import torch 
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy_topk(output, target, k=1):
    """Computes the topk accuracy"""
    batch_size = target.size(0)

    _, pred = torch.topk(output, k=k, dim=1, largest=True, sorted=True)

    res_total = 0
    for curr_k in range(k):
      curr_ind = pred[:,curr_k]
      num_eq = torch.eq(curr_ind, target).sum()
      acc = num_eq/len(output)
      res_total += acc
    return res_total*100

def train(train_loader, model, criterion, optimizer, epoch, device, print_freq=500):
    '''
    Run one train epoch
    '''
    losses = AverageMeter()
    accs = AverageMeter()

    # switch to train mode
    model.train()

    for i, (x, target) in enumerate(train_loader):

        x = x.to(device)
        target = target.to(device)

        # Forward pass
        logits = model(x)
        loss = criterion(logits, target)

        # Backward pass and update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        acc = accuracy_topk(logits.data, target)
        accs.update(acc.item(), x.size(0))
        losses.update(loss.item(), x.size(0))

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Accuracy {prec.val:.3f} ({prec.avg:.3f})'.format(
                      epoch, i, len(train_loader),
                      loss=losses, prec=accs))

@torch.no_grad()
def eval(val_loader, model, criterion, device):
    '''
    Run evaluation
    '''
    losses = AverageMeter()
    accs = AverageMeter()

    # switch to eval mode
    model.eval()


    for i, (x, target) in enumerate(val_loader):

        x = x.to(device)
        target = target.to(device)

        # Forward pass
        logits = model(x)
        loss = criterion(logits, target)

        # measure accuracy and record loss
        acc = accuracy_topk(logits.data, target)
        accs.update(acc.item(), x.size(0))
        losses.update(loss.item(), x.size(0))

    print('Dev in\t Loss ({loss.avg:.4f})\t'
            'Accuracy ({prec.avg:.3f})\n'.format(
              loss=losses, prec=accs))

class ResNetBlock(nn.Module):
    '''
    Residual Block
    '''
    def __init__(self, num_feats, layer_size, dropout_p):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(num_features=num_feats),
            nn.Linear(num_feats, layer_size),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(layer_size, num_feats),
            nn.Dropout(dropout_p)
        )
    def forward(self, x):
        return x + self.block(x)
    

class ResNet(nn.Module):
    '''
        ResNet
    '''
    def __init__(self, num_feats=123, num_classes=9):
        super().__init__()
        
        layer_size=256
        dropout_p = 0.25
        
        self.predictor = nn.Sequential(
            ResNetBlock(num_feats, layer_size, dropout_p),
            ResNetBlock(num_feats, layer_size, dropout_p),
            ResNetBlock(num_feats, layer_size, dropout_p),
            ResNetBlock(num_feats, layer_size, dropout_p),
            ResNetBlock(num_feats, layer_size, dropout_p),
            ResNetBlock(num_feats, layer_size, dropout_p),
            ResNetBlock(num_feats, layer_size, dropout_p),
            nn.BatchNorm1d(num_features=num_feats),
            nn.ReLU(),
            nn.Linear(num_feats, num_classes)
        )
    def forward(self, x):
        return self.predictor(x)


def get_default_device():
    if torch.cuda.is_available():
        print("Got CUDA!")
        return torch.device('cuda')
    else:
        print("No CUDA found")
        return torch.device('cpu')

def get_lab_to_ind(data_df):
    '''
    Prepare a label to index map
    '''
    y_fact = set(list(data_df['fact_cwsm_class']))
    lab_to_ind = {}
    for i, lab in enumerate(y_fact):
        lab_to_ind[lab] = i
    return lab_to_ind

def normalize(
    X: Dict[str, np.ndarray], normalization: str, seed: int, noise: float = 1e-3
) -> Dict[str, np.ndarray]:
    # X ~ {'train': <train_size x n_features>, 'val': <val_size x n_features>, 'test': <test_size x n_features>}
    X_train = X['train']
    if normalization == 'standard':
        normalizer = sklearn.preprocessing.StandardScaler()
    elif normalization == 'quantile':
        normalizer = sklearn.preprocessing.QuantileTransformer(
            output_distribution='normal',
            n_quantiles=max(min(X['train'].shape[0] // 30, 1000), 10),
            subsample=1e9,
            random_state=seed,
        )
        if noise:
            X_train = X_train.copy()
            stds = np.std(X_train, axis=0, keepdims=True)
            noise_std = noise / np.maximum(stds, noise)
            X_train += noise_std * np.random.default_rng(seed).standard_normal(
                X_train.shape
            )
    else:
        raise ValueError(f'unknown normalization: {normalization}')
    normalizer.fit(X_train)
    return {k: normalizer.transform(v) for k, v in X.items()}


def main():

    parser = argparse.ArgumentParser(description='Train FTTransformer.')
    parser.add_argument('train_path', type=str, help='Path to train data')
    parser.add_argument('dev_in_path', type=str, help='Path to dev_in data')
    parser.add_argument('save_dir_path', type=str, help='Path to directory to save')
    parser.add_argument('--seed', type=int, default=1, help='Specify the global random seed')

    args = parser.parse_args()

    # Set Seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/train_resnet.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')
    
    df_train = pd.read_csv(args.train_path)
    df_dev_in = pd.read_csv(args.dev_in_path)

    # Identify the categorical features
    cat_features = []
    for col in df_dev_in:
        values = df_dev_in[col].tolist()
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
        df_dev_in[col] = df_dev_in[col].fillna(nan_replacements[col])

    # Normalise using train data stats
    # Quantile normalisation is used (maps to a normal distribution)
    X_train_np = np.asarray(df_train.iloc[:,6:])
    X_dev_in_np = np.asarray(df_dev_in.iloc[:,6:])
    X = {'train': X_train_np, 'dev_in': X_dev_in_np}
    X = normalize(X, normalization='quantile', seed=args.seed)
    X_train_np = X['train']
    X_dev_in_np = X['dev_in']

    X_train = torch.FloatTensor(X_train_np)
    X_dev_in = torch.FloatTensor(X_dev_in_np)

    lab_to_ind = get_lab_to_ind(df_train)
    batch_size = 1024

    # Train
    y_train = np.asarray(df_train['fact_cwsm_class'])
    y_train = torch.LongTensor(np.asarray([lab_to_ind[lab] for lab in y_train]))

    train_ds = TensorDataset(X_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # Dev in
    y_dev_in = df_dev_in['fact_cwsm_class']
    y_dev_in = torch.LongTensor(np.asarray([lab_to_ind[lab] for lab in y_dev_in]))

    dev_in_ds = TensorDataset(X_dev_in, y_dev_in)
    dev_in_dl = DataLoader(dev_in_ds, batch_size=batch_size, shuffle=True)


    # Get the device
    device = get_default_device()

    # Define the model
    model = ResNet()
    model.to(device)

    # optimizer
    lr = 0.0001
    weight_decay = 0.9
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # criterion
    criterion = nn.CrossEntropyLoss().to(device)

    # Train
    epochs = 30
    for epoch in range(epochs):

        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(train_dl, model, criterion, optimizer, epoch, device)

        # evaluate on validation set
        eval(dev_in_dl, model, criterion, device)
    
    state = model.state_dict()
    torch.save(state, f'{args.save_dir_path}/seed{args.seed}.th')


if __name__ == '__main__':
    main()