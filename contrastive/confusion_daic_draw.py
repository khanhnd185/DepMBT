from copy import deepcopy
import os
import torch
import pickle
import argparse
from tqdm import tqdm
from dataset import collate_fn
from daic import DAICWOZ
from utils import *
from model import SupConMBT, CEMBT, EarlyConcat, MS2OS, CrossAttention, FullAttention
from torch.utils.data import DataLoader
from sklearn.metrics import recall_score, precision_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, plot_confusion_matrix
import matplotlib.pyplot as plt

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
    all_labels, all_y = transform(all_labels, all_y)

    f1 = f1_score(all_labels, all_y, average='weighted')
    r = recall_score(all_labels, all_y, average='weighted')
    p = precision_score(all_labels, all_y, average='weighted')
    acc = accuracy_score(all_labels, all_y)
    cm = confusion_matrix(all_labels, all_y)
    return [cm, f1, r, p, acc, cm]

def val(net, validldr, softmax, temp):
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
    all_labels, all_y = transform(all_labels, all_y)

    f1 = f1_score(all_labels, all_y, average='weighted')
    r = recall_score(all_labels, all_y, average='weighted')
    p = precision_score(all_labels, all_y, average='weighted')
    acc = accuracy_score(all_labels, all_y)
    cm = confusion_matrix(all_labels, all_y)
    return [cm, f1, r, p, acc, cm]

def main():
    parser = argparse.ArgumentParser(description='Train task seperately')

    parser.add_argument('--batch', '-b', type=int, default=32, help='Batch size')
    parser.add_argument('--temp', '-t', type=float, default=1.0, help='Softmax temperature')
    parser.add_argument('--rate', '-R', default='4', help='Rate')
    parser.add_argument('--datadir', '-d', default='../../../../Data/DAICWoz/', help='Data folder path')

    args = parser.parse_args()

    title_size = 16
    colorbar = False
    plt.rcParams.update({'font.size':16})
    cmap = "Reds"
    display_labels = ['normal', 'depressed']
    values_format = "2d"

    f, axes = plt.subplots(2, 3, figsize=(10, 16))

    with open('confusion_matrix.pickle', 'rb') as handle:
        confusion_matrices = pickle.load(handle)

    ConfusionMatrixDisplay(confusion_matrix=confusion_matrices['EC'], display_labels=display_labels).plot(
        include_values=True, cmap=cmap, ax=axes[0, 0], colorbar=colorbar, values_format=values_format)
    axes[0, 0].xaxis.set_ticklabels(['', ''])
    axes[0, 0].set_xlabel('')
    axes[0, 0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    axes[0, 1].set_title("HA", size=title_size)
    ConfusionMatrixDisplay(confusion_matrix=confusion_matrices['HA'], display_labels=display_labels).plot(
        include_values=True, cmap=cmap, ax=axes[0, 1], colorbar=colorbar, values_format=values_format)
    axes[0, 1].xaxis.set_ticklabels(['', ''])
    axes[0, 1].yaxis.set_ticklabels(['', ''])
    axes[0, 1].set_xlabel('')
    axes[0, 1].set_ylabel('')
    axes[0, 1].tick_params(axis='both', which='both', bottom=False, left=False)


    axes[1, 0].set_title("DepMBT-CE", size=title_size)
    ConfusionMatrixDisplay(confusion_matrix=confusion_matrices['CEMBT'], display_labels=display_labels).plot(
        include_values=True, cmap=cmap, ax=axes[1, 0], colorbar=colorbar, values_format=values_format)
    axes[1, 0].tick_params(axis='x', which='both', bottom=False)

    axes[0, 2].set_title("CAC", size=title_size)
    ConfusionMatrixDisplay(confusion_matrix=confusion_matrices['CAC'], display_labels=display_labels).plot(
        include_values=True, cmap=cmap, ax=axes[0, 2], colorbar=colorbar, values_format=values_format)
    axes[0, 2].yaxis.set_ticklabels(['', ''])
    axes[0, 2].set_ylabel('')
    axes[0, 2].tick_params(axis='y', which='both',left=False)


    axes[1, 1].set_title("DepMBT-SC", size=title_size)
    ConfusionMatrixDisplay(confusion_matrix=confusion_matrices['SCMBT'], display_labels=display_labels).plot(
        include_values=True, cmap=cmap, ax=axes[1, 1], colorbar=colorbar, values_format=values_format)
    axes[1, 1].yaxis.set_ticklabels(['', ''])
    axes[1, 1].set_ylabel('')
    axes[1, 1].tick_params(axis='both', which='both',bottom=False, left=False)

    axes[-1, -1].axis('off')


    f.suptitle("Confusion matrices for Models with DAIC-WOZ dataset", size=title_size, y=0.93)

    plt.legend()
    plt.show()


if __name__=="__main__":
    main()
