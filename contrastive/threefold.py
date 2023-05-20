from copy import deepcopy
import os
import pickle
import argparse
from poplib import CR
from tqdm import tqdm

import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score, precision_score, accuracy_score, confusion_matrix

from utils import *
from eatd import EATD
from model import CEMBT, CrossAttention, FullAttention

def train(net, trainldr, optimizer, epoch, epochs, learning_rate, criteria):
    total_losses = AverageMeter()
    net.train()

    train_loader_len = len(trainldr)
    for batch_idx, data in enumerate(tqdm(trainldr)):
        feature_audio, feature_video, mask, labels = data

        adjust_learning_rate(optimizer, epoch, epochs, learning_rate, batch_idx, train_loader_len)
        feature_audio = feature_audio.cuda()
        feature_video = feature_video.cuda()
        mask = mask.cuda()
        labels = labels.long()
        labels = labels.cuda()

        output = net(feature_audio, feature_video, mask)
        loss = criteria(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_losses.update(loss.data.item(), feature_audio.size(0))
    return total_losses.avg()

def transform(y, yhat):
    i = np.argmax(yhat, axis=1)
    yhat = np.zeros(yhat.shape)
    yhat[np.arange(len(i)), i] = 1

    if not len(y.shape) == 1:
        if y.shape[1] == 1:
            y = y.reshape(-1)
        else:
            y = np.argmax(y, axis=-1)
    if not len(yhat.shape) == 1:
        if yhat.shape[1] == 1:
            yhat = yhat.reshape(-1)
        else:
            yhat = np.argmax(yhat, axis=-1)

    return y, yhat

def val(net, validldr, criteria):
    total_losses = AverageMeter()
    net.eval()

    all_y = None
    all_labels = None
    for batch_idx, data in enumerate(tqdm(validldr)):
        feature_audio, feature_video, mask, labels = data
        with torch.no_grad():
            feature_audio = feature_audio.cuda()
            feature_video = feature_video.cuda()
            mask = mask.cuda()
            labels = labels.long()
            labels = labels.cuda()

            y = net(feature_audio, feature_video, mask)
            loss = criteria(y, labels)
            total_losses.update(loss.data.item(), feature_audio.size(0))

            if all_y == None:
                all_y = y.clone()
                all_labels = labels.clone()
            else:
                all_y = torch.cat((all_y, y), 0)
                all_labels = torch.cat((all_labels, labels), 0)

    all_y = all_y.cpu().numpy()
    all_labels = all_labels.cpu().numpy()
    all_labels, all_y = transform(all_labels, all_y)

    f1 = f1_score(all_labels, all_y)
    r = recall_score(all_labels, all_y)
    p = precision_score(all_labels, all_y)
    acc = accuracy_score(all_labels, all_y)
    cm = confusion_matrix(all_labels, all_y)
    return (total_losses.avg(), f1, r, p, acc, cm)


def main():
    parser = argparse.ArgumentParser(description='Generate dataset')
    parser.add_argument('--datadir', '-d', default='../../../../Data/EATD-Corpus/', help='Data folder path')
    
    parser.add_argument('--fold', default='results/EATD0/train_indexes.pickle', help='Config number')
    parser.add_argument('--config', '-c', type=int, default=7, help='Config number')
    parser.add_argument('--batch', '-b', type=int, default=16, help='Batch size')
    parser.add_argument('--opt', '-o', default='adam', help='Optimizer')
    parser.add_argument('--epoch', '-e', type=int, default=10, help='Number of epoches')
    parser.add_argument('--lr', '-a', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--loss', '-l', default='focal', help='Loss function')
    args = parser.parse_args()
    output_dir = 'EATD{}'.format(str(args.config))

    x_text = np.load(args.datadir+'whole_samples_reg_avg.npz')['arr_0']
    y_text = np.load(args.datadir+'whole_labels_reg_avg.npz')['arr_0']
    x_audio = np.load(args.datadir+'whole_samples_reg_256.npz')['arr_0']
    y_audio = np.load(args.datadir+'whole_labels_reg_256.npz')['arr_0']

    y_text_cls = y_text >= 53.0
    y_text_cls = y_text_cls.astype(int)

    test_indexes = [[], [], []]
    with open(args.fold, 'rb') as handle:
        train_indexes = pickle.load(handle)

    for f in range(3):
        for i in range(len(y_text)):
            if i not in train_indexes[f]:
                test_indexes[f].append(i)

    if args.loss == 'focal':
        train_criteria = FocalLoss(gamma=1.0)
        valid_criteria = FocalLoss(gamma=1.0)
    else:
        train_criteria = nn.CrossEntropyLoss()
        valid_criteria = nn.CrossEntropyLoss()

    df = create_new_df()
    df['fold'] = []


    test_dataset = []
    test_dataldr = []
    train_dataset = []
    train_dataldr = []
    train_permulation = [1,2,3,4,5]
    test_permulation = [1,2,3]
    for fold in range(3):
        train_dataset.append(EATD(x_audio, x_text, y_text, train_indexes[fold], train_permulation))
        train_dataldr.append(DataLoader(train_dataset[fold], batch_size=16, shuffle=True, num_workers=0))
        print(len(train_dataset[fold]))
        print(np.sum(train_dataset[fold].label))

        test_dataset.append(EATD(x_audio, x_text, y_text, test_indexes[fold], test_permulation))
        test_dataldr.append(DataLoader(test_dataset[fold], batch_size=16, shuffle=False, num_workers=0))
        print(len(test_dataset[fold]))
        print(np.sum(test_dataset[fold].label))

    best_f1 = [0.0] * 3
    save_precision = [0.0] * 3
    save_recall = [0.0] * 3

    for f in range(3):
        net = CEMBT(1024, 256 , 32, head='mlp', project_type='conv1d', feed_forward=32, num_layers=4, num_bottle_token=1, drop=0.2)
        net = net.cuda()

        if args.opt == 'SGD':
            optimizer = torch.optim.SGD(net.parameters(), momentum=0.9, lr=args.lr, weight_decay=1.0/args.batch)
        else:
            optimizer = torch.optim.AdamW(net.parameters(), betas=(0.9, 0.999), lr=args.lr, weight_decay=1.0/args.batch)

        for epoch in range(args.epoch):
            train_loss = train(net, train_dataldr[f], optimizer, epoch, args.epoch, args.lr, train_criteria)

            eval_return = val(net, test_dataldr[f], valid_criteria)
            _, val_f1, val_recall, val_precision, _, _ = eval_return
            description = "Epoch {:2d} | Fold {} | Trainloss {:.5f}:".format(epoch, f, train_loss)
            print_eval_info(description, eval_return)

            os.makedirs(os.path.join('results', output_dir), exist_ok = True)

            if val_f1 >= best_f1[f]:
                checkpoint = {'state_dict': net.state_dict()}
                torch.save(checkpoint, os.path.join('results', output_dir, 'best_fold{}.pth'.format(f)))
                best_f1[f] = val_f1
                save_recall[f] = val_recall
                save_precision[f] = val_precision

            df = append_entry_df(df, eval_return)
            df['fold'].append(f)

    for f in range(3):
        print("Fold {} best f1-precision-recall: {:.5f},{:.5f},{:.5f}".format(f, best_f1[f], save_precision[f], save_recall[f]))
    
    
    with open(os.path.join('results', output_dir, 'train_indexes.pickle'), 'wb') as handle:
        pickle.dump(train_indexes, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__=="__main__":
    main()

