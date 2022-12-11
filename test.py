import torch
import argparse
from tqdm import tqdm
from data import DVlog, collate_fn
from helpers import *
from models import FeatureFusion, StanfordTransformerFusion
from torch.utils.data import DataLoader

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
    parser.add_argument('--datadir', '-d', default='../../../Data/DVlog/', help='Data folder path')
    args = parser.parse_args()

    validset = DVlog(args.datadir+'test'+args.rate+'.pickle')
    valid_criteria = nn.BCELoss()
    validldr = DataLoader(validset, batch_size=args.batch, collate_fn=collate_fn, shuffle=False, num_workers=0)

    if args.net == "AnnotatedTrasformer":
        net = StanfordTransformerFusion(136, 25, 128)
    else:
        net = FeatureFusion(161, hidden_features=1024, out_features=1)

    if args.resume != '':
        print("Resume form | {} ]".format(args.resume))
        net = load_state_dict(net, args.resume)
    else:
        print("No input")
        return

    net = nn.DataParallel(net).cuda()
    val_loss, val_metrics = val(net, validldr, valid_criteria)
    print('Downrate {}: {:.5f},{:.5f}'.format(args.rate, val_loss, val_metrics))

if __name__=="__main__":
    main()

