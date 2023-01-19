import argparse
import torch
from model import CEMBT, EarlyConcat, MS2OS
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

    parser.add_argument('--net', '-n', default='ms2os', help='Net name')
    args = parser.parse_args()

    if args.net == 'early':
        model = EarlyConcat(136, 25, 256)
    elif args.net == 'ms2os':
        model = MS2OS(136, 25, 256)
    else:
        model = CEMBT(136, 25 , 256, head='mlp')
    input_shape = (16, 512)
    feature_sizes = (25, 136)
    flops_count, params_count = get_model_complexity_info(
            model, input_shape, as_strings=True,
            input_constructor=partial(bert_input_constructor, feature_sizes=feature_sizes),
            print_per_layer_stat=False)
    print("Model " + args.net)
    print('{:<30}  {:<8}'.format('Computational complexity: ', flops_count))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params_count))
