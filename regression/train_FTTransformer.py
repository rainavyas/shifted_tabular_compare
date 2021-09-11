import rtdl
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import argparse
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from scipy import stats
from typing import Dict
import numpy as np
import sklearn.preprocessing


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



def apply_model(model, x_num, x_cat=None):
    '''
    FTTransformer expects numerical and categorical inputs separately
    '''
    return model(x_num, x_cat) if isinstance(model, rtdl.FTTransformer) else model(x_num)

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

def train(train_loader, model, criterion, optimizer, epoch, device, print_freq=2000):
    '''
    Run one train epoch
    '''
    losses = AverageMeter()
    rmses = AverageMeter()

    # switch to train mode
    model.train()

    for i, (x, target) in enumerate(train_loader):

        x = x.to(device)
        target = target.to(device)

        # Forward pass
        out = apply_model(model, x)
        means = out[:,0]
        variances = out[:,1]**2
        loss = criterion(means, target, variances)

        # Backward pass and update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure rmse and record loss
        mse_func = nn.MSELoss()
        rmse = torch.sqrt(mse_func(means.data, target))
        rmses.update(rmse.item(), x.size(0))
        losses.update(loss.item(), x.size(0))

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'RMSE {prec.val:.3f} ({prec.avg:.3f})'.format(
                      epoch, i, len(train_loader),
                      loss=losses, prec=rmses))

@torch.no_grad()
def eval(val_loader, model, criterion, device):
    '''
    Run evaluation
    '''
    losses = AverageMeter()
    rmses = AverageMeter()

    # switch to eval mode
    model.eval()


    for i, (x, target) in enumerate(val_loader):

        x = x.to(device)
        target = target.to(device)

        # Forward pass
        out = apply_model(model, x)
        means = out[:,0]
        variances = out[:,1]**2
        loss = criterion(means, target, variances)

        # measure rmse and record loss
        mse_func = nn.MSELoss()
        rmse = torch.sqrt(mse_func(means.data, target))
        rmses.update(rmse.item(), x.size(0))
        losses.update(loss.item(), x.size(0))

    print('Dev in\t Loss ({loss.avg:.4f})\t'
            'RMSE ({prec.avg:.3f})\n'.format(
              loss=losses, prec=rmses))



def get_default_device():
    if torch.cuda.is_available():
        print("Got CUDA!")
        return torch.device('cuda')
    else:
        print("No CUDA found")
        return torch.device('cpu')

def main():

    parser = argparse.ArgumentParser(description='Train FTTransformer.')
    parser.add_argument('train_path', type=str, help='Path to train data')
    parser.add_argument('dev_in_path', type=str, help='Path to dev_in data')
    parser.add_argument('--epochs', type=int, default=100, help='Specify the number of epochs to train for')
    parser.add_argument('--seed', type=int, default=1, help='Specify the global random seed')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('--save_dir', type=str, help='Load path to which trained model will be saved')

    args = parser.parse_args()

    df_train = pd.read_csv(args.train_path)
    df_dev_in = pd.read_csv(args.dev_in_path)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    seed = args.seed

    batch_size = args.batch_size


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
            # print(nan_replacements[col])
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
    X = normalize(X, normalization='quantile', seed=seed)
    X_train_np = X['train']
    X_dev_in_np = X['dev_in']

    X_train = torch.FloatTensor(X_train_np)
    X_dev_in = torch.FloatTensor(X_dev_in_np)


    # Train
    y_train = np.asarray(df_train['fact_temperature'])
    y_train = torch.LongTensor(y_train)

    train_ds = TensorDataset(X_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # Dev in
    y_dev_in = df_dev_in['fact_temperature']
    y_dev_in = torch.LongTensor(y_dev_in)

    dev_in_ds = TensorDataset(X_dev_in, y_dev_in)
    dev_in_dl = DataLoader(dev_in_ds, batch_size=batch_size, shuffle=True)


    device = get_default_device()

    # Create the Feature Transformer Model (mean and variance output)

    model = rtdl.FTTransformer.make_default(
        n_num_features=X_train.shape[1],
        cat_cardinalities=None,
        last_layer_query_idx=[-1],
        d_out=2
    )

    model.to(device)

    optimizer = (
        model.make_default_optimizer()
        if isinstance(model, rtdl.FTTransformer)
        else torch.optim.AdamW(model.parameters(), lr=lr,
                              weight_decay=weight_decay)
    )



    # create loss function criterion
    criterion = nn.GaussianNLLLoss().to(device)


    # Train
    epochs = args.epochs
    for epoch in range(epochs):

        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(train_dl, model, criterion, optimizer, epoch, device)

        # evaluate on validation set
        eval(dev_in_dl, model, criterion, device)




    state = model.state_dict()
    torch.save(state, f'{args.save_dir}/model{args.seed}.th')




if __name__ == '__main__':
    main()
