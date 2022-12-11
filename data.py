import os
import torch
import pickle
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


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


def gen_dataset(rate):
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
        audio = audio[::rate, :]
        visual = visual[::rate, :]

        dataset[fold].append((audio, visual, label_index[label]))

    for fold in dataset.keys():
        with open(os.path.join(data_dir, fold+str(rate)+'.pickle'), 'wb') as handle:
            pickle.dump(dataset[fold], handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__=="__main__":
    downsample_rate = 1
    gen_dataset(downsample_rate)
