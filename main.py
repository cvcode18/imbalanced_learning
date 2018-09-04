import argparse
import mxnet as mx
import os
from wider_training import train
from wider_testing import test

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deep Imbalanced Classification')
    parser.add_argument('--data_path', help='data directory')
    parser.add_argument('--epochs', default=250, type=int, help='epochs')
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--wd', default=0.0005, type=float, help='weight decay')
    parser.add_argument('--mom', default=0.9, type=float, help='momentum')
    parser.add_argument('--batch_size', default=24, type=int, help='batch size')
    parser.add_argument('--num_classes', default=14, type=int, help='number of classes')
    parser.add_argument('--finetune', action='store_true', help='fine tune backbone architecture or not?')
    parser.add_argument('--test', action='store_true', help='testing')

    args = parser.parse_args()

    # Parameter Naming
    params_name = 'saved_models/base_resNet.params'

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    ctx = mx.gpu()

    if args.test:
        test(args, ctx)
    else:
        train(args, ctx)

