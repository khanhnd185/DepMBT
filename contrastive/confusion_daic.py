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

    with open('daic.pickle', 'rb') as handle:
        dataset = pickle.load(handle)
    testset = DAICWOZ(dataset, args.datadir + 'test.csv', is_train=False)
    test_criteria = nn.Softmax(dim=1)
    testldr = DataLoader(testset, batch_size=args.batch, collate_fn=collate_fn, shuffle=False, num_workers=0)

    confusion_matrices = {}
    ### Model 1
    net = EarlyConcat(136, 25, 256)
    net = load_state_dict(net, 'results/DAIC-CEearly-mlp04/best_val_f1.pth')
    net = net.cuda()
    eval_return = val(net, testldr, test_criteria, args.temp)
    axes[0, 0].set_title("EC", size=title_size)
    confusion_matrices['EC'] = eval_return[0]
    ConfusionMatrixDisplay(confusion_matrix=eval_return[0], display_labels=display_labels).plot(
        include_values=True, cmap=cmap, ax=axes[0, 0], colorbar=colorbar, values_format=values_format)
    axes[0, 0].xaxis.set_ticklabels(['', ''])
    axes[0, 0].set_xlabel('')
    axes[0, 0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    eval_return[0] = 0
    description = 'EC'
    print_eval_info(description, eval_return)
    

    ### Model 2
    net = MS2OS(136, 25, 256)
    net = load_state_dict(net, 'results/DAIC-CEms2os-mlp02/best_val_f1.pth')
    net = net.cuda()
    eval_return = val(net, testldr, test_criteria, args.temp)
    confusion_matrices['HA'] = eval_return[0]

    axes[0, 1].set_title("HA", size=title_size)
    ConfusionMatrixDisplay(confusion_matrix=eval_return[0], display_labels=display_labels).plot(
        include_values=True, cmap=cmap, ax=axes[0, 1], colorbar=colorbar, values_format=values_format)
    axes[0, 1].xaxis.set_ticklabels(['', ''])
    axes[0, 1].yaxis.set_ticklabels(['', ''])
    axes[0, 1].set_xlabel('')
    axes[0, 1].set_ylabel('')
    axes[0, 1].tick_params(axis='both', which='both', bottom=False, left=False)
    eval_return[0] = 0
    description = 'HA'
    print_eval_info(description, eval_return)


    ### Model 3
    net = CEMBT(136, 25 , 256, head='mlp')
    net = load_state_dict(net, 'results/DAIC-CEmbt-mlp08/best_val_f1.pth')
    net = net.cuda()
    eval_return = val(net, testldr, test_criteria, args.temp)
    confusion_matrices['CEMBT'] = eval_return[0]

    axes[1, 0].set_title("DepMBT-CE", size=title_size)
    ConfusionMatrixDisplay(confusion_matrix=eval_return[0], display_labels=display_labels).plot(
        include_values=True, cmap=cmap, ax=axes[1, 0], colorbar=colorbar, values_format=values_format)
    axes[1, 0].tick_params(axis='x', which='both', bottom=False)
    eval_return[0] = 0
    description = 'DepMBT-CE'
    print_eval_info(description, eval_return)


    # Model 5
    net = CrossAttention(136, 25, 256)
    net = load_state_dict(net, 'results/DAIC-CEcross-mlp01/best_val_f1.pth')
    net = net.cuda()
    eval_return = val(net, testldr, test_criteria, args.temp)
    confusion_matrices['CAC'] = eval_return[0]

    axes[0, 2].set_title("CAC", size=title_size)
    ConfusionMatrixDisplay(confusion_matrix=eval_return[0], display_labels=display_labels).plot(
        include_values=True, cmap=cmap, ax=axes[0, 2], colorbar=colorbar, values_format=values_format)
    axes[0, 2].yaxis.set_ticklabels(['', ''])
    axes[0, 2].set_ylabel('')
    axes[0, 2].tick_params(axis='y', which='both',left=False)
    eval_return[0] = 0
    description = 'CAC'
    print_eval_info(description, eval_return)

    axes[-1, -1].axis('off')


    ### Model 4
    testset = DAICWOZ(dataset, args.datadir + 'test.csv', is_train=False, maxlen=768)
    test_criteria = nn.Softmax(dim=1)
    testldr = DataLoader(testset, batch_size=args.batch, collate_fn=collate_fn, shuffle=False, num_workers=0)
    net = SupConMBT(136, 25 , 256)
    classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, 2)
        )
    net = load_state_dict(net, 'results/DAIC-SupConMBT-06/best0.pth')
    classifier = load_state_dict(classifier, 'results/DAIC-SupConHead-mlp-06/best_val_f1.pth')
    net = net.cuda()
    classifier = classifier.cuda()
    eval_return = val_supcon(net, classifier, testldr, test_criteria, args.temp)
    confusion_matrices['SCMBT'] = eval_return[0]

    axes[1, 1].set_title("DepMBT-SC", size=title_size)
    ConfusionMatrixDisplay(confusion_matrix=eval_return[0], display_labels=display_labels).plot(
        include_values=True, cmap=cmap, ax=axes[1, 1], colorbar=colorbar, values_format=values_format)
    axes[1, 1].yaxis.set_ticklabels(['', ''])
    axes[1, 1].set_ylabel('')
    axes[1, 1].tick_params(axis='both', which='both',bottom=False, left=False)
    eval_return[0] = 0
    description = 'DepMBT-SC'
    print_eval_info(description, eval_return)

    f.suptitle("Confusion matrices for Models with DAIC-WOZ dataset", size=title_size, y=0.93)

    #plt.legend()
    #plt.show()
    
    with open(('confusion_matrix.pickle'), 'wb') as handle:
        pickle.dump(confusion_matrices, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__=="__main__":
    main()
