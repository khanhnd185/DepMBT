from copy import deepcopy
import os
import torch
import pandas
import argparse
from tqdm import tqdm
from dataset import MultiViewDVlog, multiview_collate_fn
from loss import SupConLoss
from utils import *
from model import SupConMBT
from torch.utils.data import DataLoader


def train(net, trainldr, optimizer, epoch, epochs, learning_rate, criteria):
    total_losses = AverageMeter()
    net.train()
    train_loader_len = len(trainldr)
    for batch_idx, data in enumerate(tqdm(trainldr)):
        feature_audio, feature_video, mask, labels = data
        feature_audio = torch.cat([feature_audio[0], feature_audio[1]], dim=0)
        feature_video = torch.cat([feature_video[0], feature_video[1]], dim=0)
        bsz = labels.shape[0]

        # adjust_learning_rate(optimizer, epoch, epochs, learning_rate, batch_idx, train_loader_len)
        feature_audio = feature_audio.cuda()
        feature_video = feature_video.cuda()
        mask = mask.cuda()
        #labels = labels.float()
        labels = labels.cuda()

        features = net(feature_audio, feature_video, mask)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        loss = criteria(features, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_losses.update(loss.data.item(), feature_audio.size(0))
    return total_losses.avg()

def main():
    parser = argparse.ArgumentParser(description='Train task seperately')

    parser.add_argument('--config', '-c', type=int, default=7, help='Config number')
    parser.add_argument('--batch', '-b', type=int, default=32, help='Batch size')
    parser.add_argument('--rate', '-R', default='4', help='Rate')
    parser.add_argument('--opt', '-o', default='adam', help='Optimizer')
    parser.add_argument('--project', '-p', default='minimal', help='projection type')
    parser.add_argument('--epoch', '-e', type=int, default=10, help='Number of epoches')
    parser.add_argument('--temp', '-t', type=float, default=0.1, help='Temperature')
    parser.add_argument('--lr', '-a', type=float, default=0.00003, help='Learning rate')
    parser.add_argument('--datadir', '-d', default='../../../../Data/DVlog/', help='Data folder path')
    parser.add_argument('--prenorm', '-P', action='store_true', help='Pre-norm')

    args = parser.parse_args()
    output_dir = 'SupConMBT{}_{}'.format(str(args.config), args.rate)
    os.makedirs(os.path.join('results', output_dir), exist_ok = True)

    train_criteria = SupConLoss(temperature=args.temp)

    trainset = MultiViewDVlog('{}MultiViewTrain_{}.pickle'.format(args.datadir, args.rate))
    trainldr = DataLoader(trainset, batch_size=args.batch, collate_fn=multiview_collate_fn, shuffle=True, num_workers=0)

    net = SupConMBT(136, 25 , 256)
    net = nn.DataParallel(net).cuda()

    if args.opt == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), momentum=0.9, lr=args.lr, weight_decay=0.5/args.batch)
    else:
        optimizer = torch.optim.AdamW(net.parameters(), betas=(0.9, 0.999), lr=args.lr, weight_decay=0.5/args.batch)

    df = create_new_df()
    best_loss = 1000.0

    for epoch in range(args.epoch):
        train_loss = train(net, trainldr, optimizer, epoch, args.epoch, args.lr, train_criteria)
        print("Epoch {:2d} | Rate {} | Trainloss {:.5f}:".format(epoch, args.rate, train_loss))

        if train_loss <= best_loss:
            checkpoint = {'state_dict': net.state_dict()}
            torch.save(checkpoint, os.path.join('results', output_dir, 'best.pth'))
            best_loss = train_loss


if __name__=="__main__":
    main()

