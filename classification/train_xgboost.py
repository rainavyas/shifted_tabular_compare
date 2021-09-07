import pandas as pd
import numpy as np
import argparse
from xgboost import XGBClassifier

def get_lab_to_ind(data_df):
    '''
    Prepare a label to index map
    '''
    y_fact = set(list(data_df['fact_cwsm_class']))
    lab_to_ind = {}
    for i, lab in enumerate(y_fact):
        lab_to_ind[lab] = i
    return lab_to_ind

def main():

    parser = argparse.ArgumentParser(description='Train FTTransformer.')
    parser.add_argument('train_path', type=str, help='Path to train data')
    parser.add_argument('dev_in_path', type=str, help='Path to dev_in data')
    parser.add_argument('save_dir_path', type=str, help='Path to directory to save')
    parser.add_argument('--seed', type=int, default=1, help='Specify the global random seed')

    args = parser.parse_args()
    SEED=args.seed

    df_train = pd.read_csv(args.train_path)
    df_dev_in = pd.read_csv(args.dev_in_path)

    lab_to_ind = get_lab_to_ind(df_train)

    # Train
    X_train = np.asarray(df_train.iloc[:,6:])
    y_train = np.asarray(df_train['fact_cwsm_class'])
    y_train = np.asarray([lab_to_ind[lab] for lab in y_train])

    # Dev in
    X_dev_in = np.asarray(df_dev_in.iloc[:,6:])
    y_dev_in = df_dev_in['fact_cwsm_class']
    y_dev_in = np.asarray([lab_to_ind[lab] for lab in y_dev_in])

    # Define the model
    params = {
        'base_score':0.5,
        'booster':'gbtree',
        'colsample_bylevel':1,
        'colsample_bynode':1,
        'colsample_bytree':1,
        'gamma':0,
        'learning_rate':0.1,
        'max_delta_step':0,
        'max_depth':6,
        'min_child_weight':1,
        'n_jobs':1,
        'objective':'multi:softprob',
        'random_state':0,
        'reg_alpha':0,
        'reg_lambda':0,
        'scale_pos_weight':1,
        'seed':SEED,
        'subsample':1,
        'verbosity':1
    }

    model = XGBClassifier(**params)

    # Train
    model.fit(X_train, y_train, eval_metric=['mlogloss', 'merror'], eval_set=[(X_train, y_train), (X_dev_in, y_dev_in)], early_stopping_rounds=25, verbose=1)

    # Save
    model.dump_model(f'{args.save_dir_path}/seed{SEED}.txt')

if __name__ == '__main__':
    main()