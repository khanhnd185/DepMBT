from copy import deepcopy
import os
import torch
import pandas
import argparse
from tqdm import tqdm
from data import DVlog, collate_fn
from sam import SAM
from helpers import *
from models import AblationModel
from torch.utils.data import DataLoader
from sklearn.metrics import recall_score, precision_score, accuracy_score, confusion_matrix


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
        #labels = labels.float()
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

    f1 = f1_score(all_labels, all_y)
    r = recall_score(all_labels, all_y)
    p = precision_score(all_labels, all_y)
    acc = accuracy_score(all_labels, all_y)
    cm = confusion_matrix(all_labels, all_y)
    return (total_losses.avg(), f1, r, p, acc, cm)

def main():
    parser = argparse.ArgumentParser(description='Train task seperately')

    parser.add_argument('--config', '-c', type=int, default=7, help='Config number')
    parser.add_argument('--batch', '-b', type=int, default=16, help='Batch size')
    parser.add_argument('--rate', '-R', default='2', help='Batch size')
    parser.add_argument('--epoch', '-e', type=int, default=10, help='Number of epoches')
    parser.add_argument('--lr', '-a', type=float, default=0.00001, help='Learning rate')
    parser.add_argument('--datadir', '-d', default='../../../Data/DVlog/', help='Data folder path')
    parser.add_argument('--sam', '-s', action='store_true', help='Apply SAM optimizer')
    parser.add_argument('--keep', '-', action='store_true', help='Keep all data in training set')

    args = parser.parse_args()
    keep = 'k' if args.keep else ''
    output_dir = 'ablation{}_{}{}'.format(str(args.config), keep, args.rate)

    train_criteria = nn.CrossEntropyLoss()
    valid_criteria = nn.CrossEntropyLoss()

    trainset = DVlog('{}train_{}{}.pickle'.format(args.datadir, keep, args.rate))
    validset = DVlog('{}valid_{}{}.pickle'.format(args.datadir, keep, args.rate))
    trainldr = DataLoader(trainset, batch_size=args.batch, collate_fn=collate_fn, shuffle=True, num_workers=0)
    validldr = DataLoader(validset, batch_size=args.batch, collate_fn=collate_fn, shuffle=False, num_workers=0)

    net = AblationModel(136, 25, 256, args.config)
    net = nn.DataParallel(net).cuda()

    if args.sam:
        base_optimizer = torch.optim.SGD
        optimizer = SAM(net.parameters(), base_optimizer, lr=args.lr, momentum=0.9, weight_decay=1.0/args.batch)
    else:
        optimizer = torch.optim.AdamW(net.parameters(), betas=(0.9, 0.999), lr=args.lr, weight_decay=1.0/args.batch)

    best_f1 = 0.0
    best_acc = 0.0
    df = create_new_df()

    for epoch in range(args.epoch):
        if args.sam:
            train_loss = train_sam(net, trainldr, optimizer, epoch, args.epoch, args.lr, train_criteria)
        else:
            train_loss = train(net, trainldr, optimizer, epoch, args.epoch, args.lr, train_criteria)

        eval_return = val(net, validldr, valid_criteria)
        _, val_f1, _, _, val_acc, _ = eval_return
        description = "Epoch {:2d} | Rate {} | Trainloss {:.5f}:".format(epoch, args.rate, train_loss)
        print_eval_info(description, eval_return)

        os.makedirs(os.path.join('results', output_dir), exist_ok = True)

        if val_f1 >= best_f1:
            checkpoint = {'state_dict': net.state_dict()}
            torch.save(checkpoint, os.path.join('results', output_dir, 'best_val_f1.pth'))
            best_f1 = val_f1
            best_f1_model = deepcopy(net)

        if val_acc >= best_acc:
            checkpoint = {'state_dict': net.state_dict()}
            torch.save(checkpoint, os.path.join('results', output_dir, 'best_val_acc.pth'))
            best_acc = val_f1
            best_acc_model = deepcopy(net)

        df = append_entry_df(df, eval_return)


    testset = DVlog('{}test_{}{}.pickle'.format(args.datadir, keep, args.rate))
    test_criteria = nn.CrossEntropyLoss()
    testldr = DataLoader(testset, batch_size=args.batch, collate_fn=collate_fn, shuffle=False, num_workers=0)

    best_f1_model = nn.DataParallel(best_f1_model).cuda()
    eval_return = val(best_f1_model, testldr, test_criteria)
    description = 'Best F1 Testset'
    print_eval_info(description, eval_return)
    df = append_entry_df(df, eval_return)

    best_acc_model = nn.DataParallel(best_acc_model).cuda()
    eval_return = val(best_acc_model, testldr, test_criteria)
    description = 'Best Acc Testset'
    print_eval_info(description, eval_return)
    df = append_entry_df(df, eval_return)


    df = pandas.DataFrame(df)
    csv_name = os.path.join('results', output_dir, 'train.csv')
    df.to_csv(csv_name)


if __name__=="__main__":
    main()

