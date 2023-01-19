from copy import deepcopy
import os
import torch
import argparse
from tqdm import tqdm
from dataset import DVlog, collate_fn
from utils import *
from model import SupConMBT, CEMBT
from torch.utils.data import DataLoader
from sklearn.metrics import recall_score, precision_score, accuracy_score, confusion_matrix


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

def val_supcon(net, classifier, validldr, criteria):
    total_losses = AverageMeter()
    net.eval()
    classifier.eval()

    all_y = None
    all_labels = None
    for batch_idx, data in enumerate(tqdm(validldr)):
        feature_audio, feature_video, mask, labels = data
        with torch.no_grad():
            feature_audio = feature_audio.cuda()
            feature_video = feature_video.cuda()
            mask = mask.cuda()
            #labels = labels.float()
            labels = labels.cuda()

            features = net.encoder(feature_audio, feature_video, mask)
            y = classifier(features.detach())
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

    f1 = f1_score(all_labels, all_y, average='weighted')
    r = recall_score(all_labels, all_y, average='weighted')
    p = precision_score(all_labels, all_y, average='weighted')
    acc = accuracy_score(all_labels, all_y)
    cm = confusion_matrix(all_labels, all_y)
    return (total_losses.avg(), f1, r, p, acc, cm)

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
            #labels = labels.float()
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

    f1 = f1_score(all_labels, all_y, average='weighted')
    r = recall_score(all_labels, all_y, average='weighted')
    p = precision_score(all_labels, all_y, average='weighted')
    acc = accuracy_score(all_labels, all_y)
    cm = confusion_matrix(all_labels, all_y)
    return (total_losses.avg(), f1, r, p, acc, cm)

def main():
    parser = argparse.ArgumentParser(description='Train task seperately')

    parser.add_argument('--net', '-n', default='mbt', help='Net name')
    parser.add_argument('--head', '-H', default='results/best_supcon/head.pth', help='Input file')
    parser.add_argument('--input', '-i', default='results/best_supcon/mbt.pth', help='Input file')
    parser.add_argument('--config', '-c', type=int, default=7, help='Config number')
    parser.add_argument('--batch', '-b', type=int, default=8, help='Batch size')
    parser.add_argument('--rate', '-R', default='4', help='Rate')
    parser.add_argument('--datadir', '-d', default='../../../../Data/DVlog/', help='Data folder path')
    parser.add_argument('--prenorm', '-P', action='store_true', help='Pre-norm')
    parser.add_argument('--keep', '-', action='store_true', help='Keep all data in training set')

    args = parser.parse_args()
    keep = 'k' if args.keep else ''


    if args.net == 'mbt':
        net = SupConMBT(136, 25 , 256)
        classifier = nn.Sequential(
                nn.Linear(256, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.2),
                nn.Linear(512, 2)
            )

        print("Resume form {} and {}]".format(args.input, args.head))
        net = load_state_dict(net, args.input)
        classifier = load_state_dict(classifier, args.head)

        net = net.cuda()
        classifier = classifier.cuda()

        testset = DVlog('{}test_{}{}.pickle'.format(args.datadir, keep, args.rate))
        test_criteria = nn.CrossEntropyLoss()
        testldr = DataLoader(testset, batch_size=args.batch, collate_fn=collate_fn, shuffle=False, num_workers=0)
        eval_return = val_supcon(net, classifier, testldr, test_criteria)
        description = 'Testset'
        print_eval_info(description, eval_return)

        testset = DVlog('{}valid_{}{}.pickle'.format(args.datadir, keep, args.rate))
        test_criteria = nn.CrossEntropyLoss()
        testldr = DataLoader(testset, batch_size=args.batch, collate_fn=collate_fn, shuffle=False, num_workers=0)
        eval_return = val_supcon(net, classifier, testldr, test_criteria)
        description = 'Valset'
        print_eval_info(description, eval_return)

        testset = DVlog('{}train_{}{}.pickle'.format(args.datadir, keep, args.rate))
        test_criteria = nn.CrossEntropyLoss()
        testldr = DataLoader(testset, batch_size=args.batch, collate_fn=collate_fn, shuffle=False, num_workers=0)
        eval_return = val_supcon(net, classifier, testldr, test_criteria)
        description = 'Trainset'
        print_eval_info(description, eval_return)
    else:
        net = CEMBT(136, 25 , 256)

        print("Resume form {}".format(args.input))
        net = load_state_dict(net, args.input)

        net = net.cuda()

        testset = DVlog('{}test_{}{}.pickle'.format(args.datadir, keep, args.rate))
        test_criteria = nn.CrossEntropyLoss()
        testldr = DataLoader(testset, batch_size=args.batch, collate_fn=collate_fn, shuffle=False, num_workers=0)
        eval_return = val(net, testldr, test_criteria)
        description = 'Testset'
        print_eval_info(description, eval_return)

        testset = DVlog('{}valid_{}{}.pickle'.format(args.datadir, keep, args.rate))
        test_criteria = nn.CrossEntropyLoss()
        testldr = DataLoader(testset, batch_size=args.batch, collate_fn=collate_fn, shuffle=False, num_workers=0)
        eval_return = val(net, testldr, test_criteria)
        description = 'Valset'
        print_eval_info(description, eval_return)

        testset = DVlog('{}train_{}{}.pickle'.format(args.datadir, keep, args.rate))
        test_criteria = nn.CrossEntropyLoss()
        testldr = DataLoader(testset, batch_size=args.batch, collate_fn=collate_fn, shuffle=False, num_workers=0)
        eval_return = val(net, testldr, test_criteria)
        description = 'Trainset'
        print_eval_info(description, eval_return)

if __name__=="__main__":
    main()
