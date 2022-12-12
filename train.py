import os
import torch
import pandas
import argparse
from tqdm import tqdm
from data import DVlog, collate_fn
from sam import SAM
from helpers import *
from models import FeatureFusion, StanfordTransformerFusion
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
        labels = labels.float()
        labels = labels.cuda()
        optimizer.zero_grad()

        y = net(feature_audio, feature_video, mask)
        loss = criteria(y, labels)
        loss.backward()
        optimizer.step()

        total_losses.update(loss.data.item(), feature_audio.size(0))
    return total_losses.avg()

def train_sam(net, trainldr, optimizer, epoch, epochs, learning_rate, criteria):
    total_losses = AverageMeter()
    net.train()
    train_loader_len = len(trainldr)
    for batch_idx, data in enumerate(tqdm(trainldr)):
        feature_audio, feature_video, mask, labels = data

        # adjust_learning_rate(optimizer, epoch, epochs, learning_rate, batch_idx, train_loader_len)
        feature_audio = feature_audio.cuda()
        feature_video = feature_video.cuda()
        mask = mask.cuda()
        labels = labels.float()
        labels = labels.cuda()
        optimizer.zero_grad()

        y = net(feature_audio, feature_video, mask)
        loss = criteria(y, labels)
        loss.backward()
        optimizer.first_step(zero_grad=True)

        y = net(feature_audio, feature_video, mask)
        criteria(y, labels).backward()
        optimizer.second_step(zero_grad=True)

        total_losses.update(loss.data.item(), feature_audio.size(0))
    return total_losses.avg()

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
            labels = labels.float()
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
    all_y = all_y >= 0.5
    all_y = all_y.long().cpu().numpy()
    all_labels = all_labels.cpu().numpy()
    metrics = f1_score(all_labels, all_y)
    return total_losses.avg(), metrics


def main():
    parser = argparse.ArgumentParser(description='Train task seperately')

    parser.add_argument('--net', '-n', default='AnnotatedTrasformer', help='Net name')
    parser.add_argument('--resume', '-r', default='', help='Input file')
    parser.add_argument('--batch', '-b', type=int, default=16, help='Batch size')
    parser.add_argument('--rate', '-R', default='2', help='Batch size')
    parser.add_argument('--epoch', '-e', type=int, default=10, help='Number of epoches')
    parser.add_argument('--lr', '-a', type=float, default=0.00001, help='Learning rate')
    parser.add_argument('--datadir', '-d', default='../../../Data/DVlog/', help='Data folder path')
    parser.add_argument('--sam', '-s', action='store_true', help='Apply SAM optimizer')

    args = parser.parse_args()
    output_dir = args.net + '_' + args.rate 

    trainset = DVlog(args.datadir+'train'+args.rate+'.pickle')
    validset = DVlog(args.datadir+'valid'+args.rate+'.pickle')
    train_criteria = nn.BCELoss()
    valid_criteria = nn.BCELoss()

    trainldr = DataLoader(trainset, batch_size=args.batch, collate_fn=collate_fn, shuffle=True, num_workers=0)
    validldr = DataLoader(validset, batch_size=args.batch, collate_fn=collate_fn, shuffle=False, num_workers=0)

    if args.net == "AnnotatedTrasformer":
        net = StanfordTransformerFusion(136, 25, 128)
    else:
        net = FeatureFusion(161, hidden_features=1024, out_features=1)
    if args.resume != '':
        print("Resume form | {} ]".format(args.resume))
        net = load_state_dict(net, args.resume)
    net = nn.DataParallel(net).cuda()

    if args.sam:
        base_optimizer = torch.optim.SGD
        optimizer = SAM(net.parameters(), base_optimizer, lr=args.lr, momentum=0.9, weight_decay=1.0/args.batch)
    else:
        optimizer = torch.optim.AdamW(net.parameters(), betas=(0.9, 0.999), lr=args.lr, weight_decay=1.0/args.batch)
    best_performance = 0.0
    epoch_from_last_improvement = 0

    df = {}
    df['epoch'] = []
    df['lr'] = []
    df['train_loss'] = []
    df['val_loss'] = []
    df['val_metrics'] = []

    for epoch in range(args.epoch):
        lr = optimizer.param_groups[0]['lr']
        if args.sam:
            train_loss = train_sam(net, trainldr, optimizer, epoch, args.epoch, args.lr, train_criteria)
        else:
            train_loss = train(net, trainldr, optimizer, epoch, args.epoch, args.lr, train_criteria)
        val_loss, val_metrics = val(net, validldr, valid_criteria)

        infostr = {'Downrate {}: {},{:.5f},{:.5f},{:.5f},{:.5f}'
                .format(args.rate,
                        epoch,
                        lr,
                        train_loss,
                        val_loss,
                        val_metrics)}
        print(infostr)

        os.makedirs(os.path.join('results', output_dir), exist_ok = True)

        if val_metrics >= best_performance:
            checkpoint = {
                'epoch': epoch,
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(checkpoint, os.path.join('results', output_dir, 'best_val_perform.pth'))
            best_performance = val_metrics
            epoch_from_last_improvement = 0
        else:
            epoch_from_last_improvement += 1

        checkpoint = {
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(checkpoint, os.path.join('results', output_dir, 'cur_model.pth'))

        df['epoch'].append(epoch)
        df['lr'].append(lr)
        df['train_loss'].append(train_loss)
        df['val_loss'].append(val_loss)
        df['val_metrics'].append(val_metrics)
   

    df = pandas.DataFrame(df)
    csv_name = os.path.join('results', output_dir, 'train.csv')
    df.to_csv(csv_name)

if __name__=="__main__":
    main()

