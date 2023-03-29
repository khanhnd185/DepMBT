import argparse
import torch
from model import CEMBT, EarlyConcat, MS2OS, CrossAttention, FullAttention
from ptflops import get_model_complexity_info

from functools import partial

def bert_input_constructor(input_shape, feature_sizes):
    bz, leng = input_shape
    leng_a, leng_v = feature_sizes
    a = torch.randn((bz, leng, leng_a))
    v = torch.randn((bz, leng, leng_v))
    m = torch.ones((bz, leng))

    return {'a':a, 'v':v, 'm':m}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train task seperately')

    parser.add_argument('--net', '-n', default='mbt', help='Net name')
    parser.add_argument('--len', '-l', type=int, default=256, help='Sequence length')
    parser.add_argument('--layer', '-L', type=int, default=4, help='Num of layers')
    args = parser.parse_args()

    if args.net == 'early':
        net = EarlyConcat(136, 25, 256)
    elif args.net == 'ms2os':
        net = MS2OS(136, 25, 256)
    elif args.net == 'cross':
        net = CrossAttention(136, 25, 256)
    elif args.net == 'full':
        net = FullAttention(136, 25, 256)
    else:
        net = CEMBT(136, 25 , 256)

    input_shape = (16, args.len)
    feature_sizes = (25, 136)
    flops_count, params_count = get_model_complexity_info(
            net, input_shape, as_strings=True,
            input_constructor=partial(bert_input_constructor, feature_sizes=feature_sizes),
            print_per_layer_stat=False)
    print("Model " + args.net)
    print('{:<30}  {:<8}'.format('Computational complexity: ', flops_count))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params_count))
