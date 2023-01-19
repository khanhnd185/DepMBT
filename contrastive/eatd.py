import os
import torch
import pickle
import itertools
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

class EATD(Dataset):
    def __init__(self, audio, text, y, fold_index, permulation_index):
        super(EATD, self).__init__()
        audio = audio[fold_index]
        audio = np.squeeze(audio)
        text = text[fold_index]
        y = y[fold_index]

        y_cls = y >= 53.0
        y_cls = y_cls.astype(int)

        for i, label in enumerate(y):
            if label < 53.0:
                continue
            for j, permulate in enumerate(itertools.permutations(audio[i], audio[i].shape[0])):
                if j not in permulation_index:
                    continue
                audio = np.vstack((audio, np.expand_dims(list(permulate), 0)))
                y_cls = np.hstack((y_cls, 1))

            for j, permulate in enumerate(itertools.permutations(text[i], text[i].shape[0])):
                if j not in permulation_index:
                    continue
                text = np.vstack((text, np.expand_dims(list(permulate), 0)))
        
        self.audio = audio
        self.text = text
        self.label = y_cls
        self.mask = np.array([1, 1, 1])

    def __len__(self):
        return len(self.audio)

    def __getitem__(self, i):
        return self.audio[i], self.text[i], self.mask, self.label[i]


if __name__=="__main__":
    from torch.utils.data import Dataset, DataLoader
    from sklearn.model_selection import StratifiedKFold

    import argparse
    parser = argparse.ArgumentParser(description='Generate dataset')
    parser.add_argument('--datadir', '-d', default='../../../../Data/EATD-Corpus/', help='Data folder path')
    args = parser.parse_args()

    x_text = np.load(args.datadir+'whole_samples_reg_avg.npz')['arr_0']
    y_text = np.load(args.datadir+'whole_labels_reg_avg.npz')['arr_0']
    x_audio = np.load(args.datadir+'whole_samples_reg_256.npz')['arr_0']
    y_audio = np.load(args.datadir+'whole_labels_reg_256.npz')['arr_0']

    y_text_cls = y_text >= 53.0
    y_text_cls = y_text_cls.astype(int)

    skf = StratifiedKFold(n_splits=3)
    train_indexes = []
    test_indexes = []
    for i, (train_index, test_index) in enumerate(skf.split(x_text, y_text_cls)):
        train_indexes.append(train_index)
        test_indexes.append(test_index)

    test_dataset = []
    test_dataldr = []
    train_dataset = []
    train_dataldr = []
    train_permulation = [1,2,3,4,5]
    test_permulation = [1,2,4,5]
    for fold in range(3):
        train_dataset.append(EATD(x_audio, x_text, y_text, train_indexes[fold], train_permulation))
        train_dataldr.append(DataLoader(train_dataset[fold], batch_size=16, shuffle=True, num_workers=0))
        print(len(train_dataset[fold]))

        test_dataset.append(EATD(x_audio, x_text, y_text, test_indexes[fold], test_permulation))
        test_dataldr.append(DataLoader(test_dataset[fold], batch_size=16, shuffle=False, num_workers=0))
        print(len(test_dataset[fold]))