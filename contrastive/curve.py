from copy import deepcopy
import os
import torch
import argparse
from tqdm import tqdm
from dataset import DVlog, collate_fn
from utils import *
from model import SupConMBT, CEMBT, EarlyConcat, MS2OS, CrossAttention, FullAttention
from torch.utils.data import DataLoader
from sklearn.metrics import recall_score, precision_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve
import matplotlib.pyplot as plt

def plot_roc_curve(true_y, y_prob, label):
    """
    plots the roc curve based of the probabilities
    """

    fpr, tpr, thresholds = roc_curve(true_y, y_prob)
    plt.plot(fpr, tpr, label=label)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

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

def val_supcon(net, classifier, validldr, softmax, temp):
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
            y = y / temp
            y = softmax(y)


            if all_y == None:
                all_y = y.clone()
                all_labels = labels.clone()
            else:
                all_y = torch.cat((all_y, y), 0)
                all_labels = torch.cat((all_labels, labels), 0)

    all_y = all_y.cpu().numpy()
    all_labels = all_labels.cpu().numpy()
    plot_roc_curve(all_labels, all_y[:,1], 'scmbt')
    all_labels, all_y = transform(all_labels, all_y)

    f1 = f1_score(all_labels, all_y, average='weighted')
    r = recall_score(all_labels, all_y, average='weighted')
    p = precision_score(all_labels, all_y, average='weighted')
    acc = accuracy_score(all_labels, all_y)
    cm = confusion_matrix(all_labels, all_y)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['normal', 'depressed'])
    disp.plot()
    plt.show()
    return (0, f1, r, p, acc, cm)

def val(net, validldr, softmax, label, temp):
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
            y = y / temp
            y = softmax(y)

            if all_y == None:
                all_y = y.clone()
                all_labels = labels.clone()
            else:
                all_y = torch.cat((all_y, y), 0)
                all_labels = torch.cat((all_labels, labels), 0)

    all_y = all_y.cpu().numpy()
    all_labels = all_labels.cpu().numpy()
    plot_roc_curve(all_labels, all_y[:,1], label)
    all_labels, all_y = transform(all_labels, all_y)

    f1 = f1_score(all_labels, all_y, average='weighted')
    r = recall_score(all_labels, all_y, average='weighted')
    p = precision_score(all_labels, all_y, average='weighted')
    acc = accuracy_score(all_labels, all_y)
    cm = confusion_matrix(all_labels, all_y)
    return (0, f1, r, p, acc, cm)

def main():
    parser = argparse.ArgumentParser(description='Train task seperately')

    parser.add_argument('--batch', '-b', type=int, default=8, help='Batch size')
    parser.add_argument('--temp', '-t', type=float, default=1.0, help='Softmax temperature')
    parser.add_argument('--rate', '-R', default='4', help='Rate')
    parser.add_argument('--datadir', '-d', default='../../../../Data/DVlog/', help='Data folder path')

    args = parser.parse_args()

    net = SupConMBT(136, 25 , 256)
    classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, 2)
        )
    net = load_state_dict(net, 'results/SupConMBT107_4/best0.pth')
    classifier = load_state_dict(classifier, 'results/SupConHead-mlp-1074/best_val_acc.pth')

    net = net.cuda()
    classifier = classifier.cuda()

    testset = DVlog('{}test_{}{}.pickle'.format(args.datadir, '', args.rate))
    test_criteria = nn.Softmax(dim=1)
    testldr = DataLoader(testset, batch_size=args.batch, collate_fn=collate_fn, shuffle=False, num_workers=0)
    eval_return = val_supcon(net, classifier, testldr, test_criteria, args.temp)
    description = 'Testset'
    print_eval_info(description, eval_return)

    for name in ['early', 'ms2os', 'cross', 'cembt']:
        if name == 'early':
            net = EarlyConcat(136, 25, 256)
            net = load_state_dict(net, 'results/CEearly-mlp19/best_val_f1.pth')
        elif name == 'ms2os':
            net = MS2OS(136, 25, 256)
            net = load_state_dict(net, 'results/CEms2os-mlp18/best_val_f1.pth')
        elif name == 'cross':
            net = CrossAttention(136, 25, 256)
            net = load_state_dict(net, 'results/CEcross-mlp03/best_val_f1.pth')
        else:
            net = CEMBT(136, 25 , 256, head='mlp')
            net = load_state_dict(net, 'results/CEmbt-mlp38/best_val_f1.pth')

        net = net.cuda()
        eval_return = val(net, testldr, test_criteria, name, args.temp)
        description = 'Testset'
        print_eval_info(description, eval_return)
    plt.legend()
    plt.show()

if __name__=="__main__":
    main()
