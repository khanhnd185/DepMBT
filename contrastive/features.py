import torch
import pickle
import argparse
from tqdm import tqdm
from dataset import DVlog, collate_fn
from utils import *
from model import SupConMBT, CEMBT
from torch.utils.data import DataLoader

def val(net, validldr):
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

            features = net.encoder(feature_audio, feature_video, mask)

            if all_y == None:
                all_y = features.clone()
                all_labels = labels.clone()
            else:
                all_y = torch.cat((all_y, features), 0)
                all_labels = torch.cat((all_labels, labels), 0)

    all_y = all_y.cpu().numpy()
    all_labels = all_labels.cpu().numpy()

    return all_y, all_labels

def main():
    parser = argparse.ArgumentParser(description='Train task seperately')

    parser.add_argument('--net', '-n', default='supcon', help='Net name')
    parser.add_argument('--output', '-o', default='feature.pickle', help='Output file')
    parser.add_argument('--input', '-i', default='results/best_supcon/mbt.pth', help='Input file')
    parser.add_argument('--batch', '-b', type=int, default=8, help='Batch size')
    parser.add_argument('--rate', '-R', default='4', help='Rate')
    parser.add_argument('--datadir', '-d', default='../../../../Data/DVlog/', help='Data folder path')
    parser.add_argument('--prenorm', '-P', action='store_true', help='Pre-norm')
    parser.add_argument('--keep', '-', action='store_true', help='Keep all data in training set')

    args = parser.parse_args()
    keep = 'k' if args.keep else ''

    if args.net == 'ce':
        net = CEMBT(136, 25, 256)
    else:
        net = SupConMBT(136, 25 , 256)

    print("Resume form {}".format(args.input))
    net = load_state_dict(net, args.input)
    net = net.cuda()

    testset = DVlog('{}test_{}{}.pickle'.format(args.datadir, keep, args.rate))
    testldr = DataLoader(testset, batch_size=args.batch, collate_fn=collate_fn, shuffle=False, num_workers=0)
    all_y, all_labels = val(net, testldr)

    testset = DVlog('{}valid_{}{}.pickle'.format(args.datadir, keep, args.rate))
    testldr = DataLoader(testset, batch_size=args.batch, collate_fn=collate_fn, shuffle=False, num_workers=0)
    y, label = val(net, testldr)
    all_y  = np.concatenate((all_y, y), axis=0)
    all_labels  = np.concatenate((all_labels, label), axis=0)

    testset = DVlog('{}train_{}{}.pickle'.format(args.datadir, keep, args.rate))
    testldr = DataLoader(testset, batch_size=args.batch, collate_fn=collate_fn, shuffle=False, num_workers=0)
    y, label = val(net, testldr)
    all_y  = np.concatenate((all_y, y), axis=0)
    all_labels  = np.concatenate((all_labels, label), axis=0)

    print(all_y.shape)
    print(all_labels.shape)

    with open(args.output, 'wb') as handle:
        pickle.dump((all_y, all_labels), handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__=="__main__":
    main()
