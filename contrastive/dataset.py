import os
import torch
import pickle
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


def masked_collate_fn(data):
    audio, video, lengtha, lengthv, labels = zip(*data)
    labels = torch.tensor(labels).long()
    lengtha = torch.tensor(lengtha).long()
    maska = torch.arange(max(lengtha))[None, :] < lengtha[:, None]
    lengthv = torch.tensor(lengthv).long()
    maskv = torch.arange(max(lengthv))[None, :] < lengthv[:, None]

    feature_audio = [torch.tensor(a).long() for a in audio]
    feature_video = [torch.tensor(v).long() for v in video]
    feature_audio = pad_sequence(feature_audio, batch_first=True, padding_value=0)
    feature_video = pad_sequence(feature_video, batch_first=True, padding_value=0)

    return feature_audio.float(), feature_video.float(), maska.long(), maskv.long(), labels



def collate_fn(data):
    audio, video, labels, lengths = zip(*data)
    labels = torch.tensor(labels).long()
    lengths = torch.tensor(lengths).long()
    mask = torch.arange(max(lengths))[None, :] < lengths[:, None]

    feature_audio = [torch.tensor(a).long() for a in audio]
    feature_video = [torch.tensor(v).long() for v in video]
    feature_audio = pad_sequence(feature_audio, batch_first=True, padding_value=0)
    feature_video = pad_sequence(feature_video, batch_first=True, padding_value=0)

    return feature_audio.float(), feature_video.float(), mask.long(), labels


def multiview_collate_fn(data):
    audio, video, labels, lengths = zip(*data)
    labels = torch.tensor(labels).long()
    lengths = lengths + lengths
    lengths = torch.tensor(lengths).long()
    mask = torch.arange(max(lengths))[None, :] < lengths[:, None]

    feature_audio = []
    feature_video = []

    for i in range(2):
        feature_audio.append([torch.tensor(a[i]).long() for a in audio])
        feature_video.append([torch.tensor(v[i]).long() for v in video])
        feature_audio[i] = pad_sequence(feature_audio[i], batch_first=True, padding_value=0).float()
        feature_video[i] = pad_sequence(feature_video[i], batch_first=True, padding_value=0).float()

    return feature_audio, feature_video, mask.long(), labels


class DVlog(Dataset):
    def __init__(self, filename):
        super(DVlog, self).__init__()
        
        with open(filename, 'rb') as handle:
            self.dataset = pickle.load(handle)
        
        self.length = [d[0].shape[0] for d in self.dataset]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx][0], self.dataset[idx][1], self.dataset[idx][2], self.length[idx]


class MultiViewDVlog(Dataset):
    def __init__(self, filename, maxlen=None):
        super(MultiViewDVlog, self).__init__()

        with open(filename, 'rb') as handle:
            dataset = pickle.load(handle)

        if maxlen == None:
            self.dataset = dataset
        else:
            self.dataset = []
            for data in dataset:
                a, v, label = data
                a[0] = a[0][:maxlen, :]
                a[1] = a[1][:maxlen, :]
                v[0] = v[0][:maxlen, :]
                v[1] = v[1][:maxlen, :]

                self.dataset.append((a, v, label))

        self.length = [d[0][0].shape[0] for d in self.dataset]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx][0], self.dataset[idx][1], self.dataset[idx][2], self.length[idx]

def trim(a, v):
    leng = min(a.shape[0], v.shape[0])
    return a[:leng, :], v[:leng, :]

def gen_multiview_dataset(rate):
    data_dir = '../../../../Data/DVlog/'
    feat_dir = os.path.join(data_dir, 'dvlog-dataset')
    label_file = os.path.join(feat_dir, 'labels.csv')
    label_index = {"depression": 1, "normal": 0}

    dataset = []

    with open(label_file, 'r', encoding='utf-8') as f:
        data_file = f.readlines()[1:]

    for i, data in tqdm(enumerate(data_file)):
        index, label, duration, gender, fold = data.strip().split(',')

        if fold != 'train':
            continue

        audio = np.load(os.path.join(feat_dir, index, index+'_acoustic.npy'))
        visual = np.load(os.path.join(feat_dir, index, index+'_visual.npy'))
        audio, visual = trim(audio, visual)

        au = []
        vi = []
        for j in range(rate):
            a = audio[j::rate, :]
            v = visual[j::rate, :]

            au.append(a)
            vi.append(v)
            # Default view is 2
            if (j % 2) == 1:
                au = trim(au[0], au[1])
                vi = trim(vi[0], vi[1])
                dataset.append((au, vi, label_index[label]))
                au = []
                vi = []

    with open(os.path.join(data_dir, 'MultiViewTrain_{}.pickle'.format(str(rate))), 'wb') as handle:
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

def gen_dataset(rate, keep):
    data_dir = '../../../Data/DVlog/'
    feat_dir = os.path.join(data_dir, 'dvlog-dataset')
    label_file = os.path.join(feat_dir, 'labels.csv')
    label_index = {"depression": 1, "normal": 0}

    dataset ={"train": [], "test": [], "valid": []}

    with open(label_file, 'r', encoding='utf-8') as f:
        data_file = f.readlines()[1:]

    for i, data in tqdm(enumerate(data_file)):
        index, label, duration, gender, fold = data.strip().split(',')

        audio = np.load(os.path.join(feat_dir, index, index+'_acoustic.npy'))
        visual = np.load(os.path.join(feat_dir, index, index+'_visual.npy'))

        leng = min(audio.shape[0], visual.shape[0])

        audio = audio[:leng, :]
        visual = visual[:leng, :]
        if fold == 'train' and keep:
            for j in range(rate):
                a = audio[j::rate, :]
                v = visual[j::rate, :]
                dataset[fold].append((a, v, label_index[label]))
        else:
            a = audio[::rate, :]
            v = visual[::rate, :]
            dataset[fold].append((a, v, label_index[label]))

    for fold in dataset.keys():
        k = 'k' if keep else ''
        with open(os.path.join(data_dir, '{}_{}{}.pickle'.format(fold, k, str(rate))), 'wb') as handle:
            pickle.dump(dataset[fold], handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate dataset')
    parser.add_argument('--rate', '-r', type=int, default=4, help='Downsample rate divisible by 2')
    parser.add_argument('--keep', '-k', action='store_true', help='Keep all data in training set')
    parser.add_argument('--multiview', '-m', action='store_true', help='Multiview dataset')
    args = parser.parse_args()

    if args.multiview:
        gen_multiview_dataset(args.rate)
    else:
        gen_dataset(args.rate, args.keep)
