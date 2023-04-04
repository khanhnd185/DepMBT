from copy import deepcopy
import os
import pickle
import argparse
from tqdm import tqdm

import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score, precision_score, accuracy_score, confusion_matrix
from dataset import masked_collate_fn

from utils import *
from eatd import EATD, NewEATD
from model import CEMBT, CEMMBT

def train(net, trainldr, optimizer, epoch, epochs, learning_rate, criteria):
    total_losses = AverageMeter()
    net.train()

    train_loader_len = len(trainldr)
    for batch_idx, data in enumerate(tqdm(trainldr)):
        feature_audio, feature_video, maska, maskv, labels = data

        # adjust_learning_rate(optimizer, epoch, epochs, learning_rate, batch_idx, train_loader_len)
        feature_audio = feature_audio.cuda()
        feature_video = feature_video.cuda()
        maska = maska.cuda()
        maskv = maskv.cuda()
        labels = labels.long()
        labels = labels.cuda()

        output = net(feature_audio, feature_video, maska, maskv)
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
    for batch_idx, data in enumerate(validldr):
        feature_audio, feature_video, maska, maskv, labels = data
        with torch.no_grad():
            feature_audio = feature_audio.cuda()
            feature_video = feature_video.cuda()
            maska = maska.cuda()
            maskv = maskv.cuda()
            labels = labels.long()
            labels = labels.cuda()

            y = net(feature_audio, feature_video, maska, maskv)
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
    # cm = standard_confusion_matrix(all_labels, all_y)
    #f1, p, r = their_evaluate(cm)
    return (total_losses.avg(), f1, r, p, acc, cm)

def standard_confusion_matrix(y_test, y_test_pred):
    """
    Make confusion matrix with format:
                  -----------
                  | TP | FP |
                  -----------
                  | FN | TN |
                  -----------
    Parameters
    ----------
    y_true : ndarray - 1D
    y_pred : ndarray - 1D

    Returns
    -------
    ndarray - 2D
    """
    [[tn, fp], [fn, tp]] = confusion_matrix(y_test, y_test_pred)
    return np.array([[tp, fp], [fn, tn]])

def their_evaluate(conf_matrix):
    precision = float(conf_matrix[0][0]) / (conf_matrix[0][0] + conf_matrix[0][1])
    recall = float(conf_matrix[0][0]) / (conf_matrix[0][0] + conf_matrix[1][0])
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score, precision, recall


def main():
    parser = argparse.ArgumentParser(description='Generate dataset')
    parser.add_argument('--datadir', '-d', default='../../../../Data/EATD-Corpus/', help='Data folder path')
    
    parser.add_argument('--config', '-c', type=int, default=7, help='Config number')
    parser.add_argument('--batch', '-b', type=int, default=16, help='Batch size')
    parser.add_argument('--opt', '-o', default='adam', help='Optimizer')
    parser.add_argument('--epoch', '-e', type=int, default=10, help='Number of epoches')
    parser.add_argument('--lr', '-a', type=float, default=0.0001, help='Learning rate')
    args = parser.parse_args()
    output_dir = 'EATD{}'.format(str(args.config))

    with open('../../../../Data/EATD-Corpus/eatd.pickle', 'rb') as handle:
        dataset = pickle.load(handle)

    y = [i['score'] for i in dataset]
    y = np.array(y)
    y = y >= 53.0

    n_samples = y.shape[0]
    skf = StratifiedKFold(n_splits=3, shuffle=True)
    train_indexes = []
    test_indexes = []

    for i, (train_index, test_index) in enumerate(skf.split(np.zeros(n_samples), y)):
        train_indexes.append(train_index)
        test_indexes.append(test_index)

    train_criteria = nn.CrossEntropyLoss()
    valid_criteria = nn.CrossEntropyLoss()
    df = create_new_df()
    df['fold'] = []


    test_dataset = []
    test_dataldr = []
    train_dataset = []
    train_dataldr = []
    audio_permulation = [0]
    text_permulation = [1,2,3,4]
    test_permulation = []
    for fold in range(3):
        train_dataset.append(NewEATD(dataset, True, y, train_indexes[fold], audio_permulation, text_permulation))
        train_dataldr.append(DataLoader(train_dataset[fold], batch_size=args.batch, shuffle=True, num_workers=0, collate_fn=masked_collate_fn))
        print(len(train_dataset[fold]))
        print(np.sum(train_dataset[fold].label))

        test_dataset.append(NewEATD(dataset, False, y, test_indexes[fold], test_permulation))
        test_dataldr.append(DataLoader(test_dataset[fold], batch_size=args.batch, shuffle=False, num_workers=0, collate_fn=masked_collate_fn))
        print(len(test_dataset[fold]))
        print(np.sum(test_dataset[fold].label))

    best_f1 = [0.0] * 3
    save_precision = [0.0] * 3
    save_recall = [0.0] * 3

    for f in range(3):
        net = CEMMBT(768, 32, 128)
        net = net.cuda()

        if args.opt == 'SGD':
            optimizer = torch.optim.SGD(net.parameters(), momentum=0.9, lr=args.lr, weight_decay=1.0/args.batch)
        else:
            optimizer = torch.optim.AdamW(net.parameters(), betas=(0.9, 0.999), lr=args.lr, weight_decay=1.0/args.batch)

        for epoch in range(args.epoch):
            train_loss = train(net, train_dataldr[f], optimizer, epoch, args.epoch, args.lr, train_criteria)

            eval_return = val(net, train_dataldr[f], valid_criteria)
            _, val_f1, val_recall, val_precision, _, _ = eval_return
            description = "Epoch {:2d} | Fold {} | Train:".format(epoch, f)
            print_eval_info(description, eval_return)

            eval_return = val(net, test_dataldr[f], valid_criteria)
            _, val_f1, val_recall, val_precision, _, _ = eval_return
            description = "Epoch {:2d} | Fold {} | Val:".format(epoch, f)
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

if __name__=="__main__":
    main()

