import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import shutil

import warnings
warnings.filterwarnings("ignore")


# Download 'FocusPath Full.zip': 'https://zenodo.org/record/3926181/files/FocusPath%20Full.zip?download=1'
    
def prepare_dataset(threshold=2):

    data_dir = './FocusPath_Full/'
    df = pd.read_csv(os.path.join(data_dir, 'DatabaseInfo.csv'))
    df['Subjective Score'] = np.abs(df['Subjective Score'])
    df['target'] = (df['Subjective Score'] <= threshold).astype('int')
    df['Name'] = df['Name'].str.replace('.tif', '.png')
    
    X1, y1 = df.drop('target', axis=1), df['target']
    X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.2, stratify=y1)
    X_train['target'] = y_train
    X_test['target'] = y_test
    
    focus_test = X_test[X_test['target'] == 1]['Name'].to_numpy().flatten()
    unfocus_test = X_test[X_test['target'] == 0]['Name'].to_numpy().flatten()
    
    X2, y2 = X_train.drop('target', axis=1), X_train['target']
    X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.25, stratify=y2)
    X_train['target'] = y_train
    X_test['target'] = y_test
    
    focus_train = X_train[X_train['target'] == 1]['Name'].to_numpy().flatten()
    unfocus_train = X_train[X_train['target'] == 0]['Name'].to_numpy().flatten()
    focus_val = X_test[X_test['target'] == 1]['Name'].to_numpy().flatten()
    unfocus_val = X_test[X_test['target'] == 0]['Name'].to_numpy().flatten()
    
    print('Dataset split:')
    print('focus_train:', len(focus_train))
    print('unfocus_train:', len(unfocus_train))
    print('focus_val:', len(focus_val))
    print('unfocus_val:', len(unfocus_val))
    print('focus_test:', len(focus_test))
    print('unfocus_test:', len(unfocus_test))
    
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')
    
    os.mkdir(train_dir)
    os.mkdir(val_dir)
    os.mkdir(test_dir)
    
    os.mkdir(os.path.join(train_dir, '1'))
    os.mkdir(os.path.join(train_dir, '0'))
    os.mkdir(os.path.join(val_dir, '1'))
    os.mkdir(os.path.join(val_dir, '0'))
    os.mkdir(os.path.join(test_dir, '1'))
    os.mkdir(os.path.join(test_dir, '0'))
    
    original_data = './FocusPath_Full/FocusPath_full/'
    
    for idx, img in enumerate(focus_train):
        src = os.path.join(original_data, img)
        dst = os.path.join(train_dir, '1')
        shutil.move(src, dst)
#         if idx % 100 == 0:
#             print(f'{idx} images moved from {src} to {dst}')
    print(f'{len(focus_train)} images moved from {src} to {dst}')
    
    for idx, img in enumerate(unfocus_train):
        src = os.path.join(original_data, img)
        dst = os.path.join(train_dir, '0')
        shutil.move(src, dst)
#         if idx % 500 == 0:
#             print(f'{idx} images moved from {src} to {dst}')
    print(f'{len(unfocus_train)} images moved from {src} to {dst}')
    
    for idx, img in enumerate(focus_val):
        src = os.path.join(original_data, img)
        dst = os.path.join(val_dir, '1')
        shutil.move(src, dst)
#         if idx % 100 == 0:
#             print(f'{idx} images moved from {src} to {dst}')
    print(f'{len(focus_val)} images moved from {src} to {dst}')
    
    for idx, img in enumerate(unfocus_val):
        src = os.path.join(original_data, img)
        dst = os.path.join(val_dir, '0')
        shutil.move(src, dst)
#         if idx % 100 == 0:
#             print(f'{idx} images moved from {src} to {dst}')
    print(f'{len(unfocus_val)} images moved from {src} to {dst}')
    
    for idx, img in enumerate(focus_test):
        src = os.path.join(original_data, img)
        dst = os.path.join(test_dir, '1')
        shutil.move(src, dst)
#         if idx % 100 == 0:
#             print(f'{idx} images moved from {src} to {dst}')
    print(f'{len(focus_test)} images moved from {src} to {dst}')
    
    for idx, img in enumerate(unfocus_test):
        src = os.path.join(original_data, img)
        dst = os.path.join(test_dir, '0')
        shutil.move(src, dst)
#         if idx % 100 == 0:
#             print(f'{idx} images moved from {src} to {dst}')
    print(f'{len(unfocus_test)} images moved from {src} to {dst}')