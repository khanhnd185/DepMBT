from copy import deepcopy
import os
import torch
import pandas
import argparse
from tqdm import tqdm
from dataset import DVlog, collate_fn
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

        # adjust_learning_rate(optimizer, epoch, epochs, learning_rate, batch_idx, train_loader_len)
        feature_audio = feature_audio.cuda()
        feature_video = feature_video.cuda()
        mask = mask.cuda()
        #labels = labels.float()
        labels = labels.cuda()
        optimizer.zero_grad()

        y = net(feature_audio, feature_video, mask)
        loss = criteria(y, labels)
        loss.backward()
        optimizer.step()

        total_losses.update(loss.data.item(), feature_audio.size(0))
    return total_losses.avg()

def main():
    parser = argparse.ArgumentParser(description='Train task seperately')

    parser.add_argument('--config', '-c', type=int, default=7, help='Config number')
    parser.add_argument('--batch', '-b', type=int, default=16, help='Batch size')
    parser.add_argument('--rate', '-R', default='4', help='Rate')
    parser.add_argument('--project', '-p', default='minimal', help='projection type')
    parser.add_argument('--epoch', '-e', type=int, default=10, help='Number of epoches')
    parser.add_argument('--temp', '-t', type=float, default=0.1, help='Temperature')
    parser.add_argument('--lr', '-a', type=float, default=0.00001, help='Learning rate')
    parser.add_argument('--datadir', '-d', default='../../../Data/DVlog/', help='Data folder path')
    parser.add_argument('--prenorm', '-P', action='store_true', help='Pre-norm')
    parser.add_argument('--keep', '-', action='store_true', help='Keep all data in training set')

    args = parser.parse_args()
    keep = 'k' if args.keep else ''
    output_dir = 'SupConMBT{}_{}{}'.format(str(args.config), keep, args.rate)

    train_criteria = SupConLoss(temperature=args.temp)

    trainset = DVlog('{}train_{}{}.pickle'.format(args.datadir, keep, args.rate))
    trainldr = DataLoader(trainset, batch_size=args.batch, collate_fn=collate_fn, shuffle=True, num_workers=0)

    net = SupConMBT(136, 25 , 256)
    net = nn.DataParallel(net).cuda()

    optimizer = torch.optim.AdamW(net.parameters(), betas=(0.9, 0.999), lr=args.lr, weight_decay=1.0/args.batch)

    df = create_new_df()

    for epoch in range(args.epoch):
        train_loss = train(net, trainldr, optimizer, epoch, args.epoch, args.lr, train_criteria)
        print("Epoch {:2d} | Rate {} | Trainloss {:.5f}:".format(epoch, args.rate, train_loss))


    torch.save({'state_dict': net.state_dict()}, os.path.join('results', output_dir, 'latest.pth'))



if __name__=="__main__":
    main()

