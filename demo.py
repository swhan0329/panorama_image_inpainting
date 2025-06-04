import argparse
from train import test
from utils.util import set_random_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./demo_data', type=str)
    parser.add_argument('--ckpt_dir', default='./checkpoint_s256', type=str)
    parser.add_argument('--result_dir', default='./demo_result', type=str)
    parser.add_argument('--log_dir', default='./demo_log', type=str)
    parser.add_argument('--network', default='PIUnet', type=str)
    parser.add_argument('--nker', default=32, type=int)
    parser.add_argument('--norm', default='bnorm', type=str)
    parser.add_argument('--seed', default=2020, type=int)
    parser.add_argument('--data_parallel', action='store_true')
    args = parser.parse_args()

    args.mode = 'test'
    args.lr = 1e-4
    args.batch_size = 1
    args.num_epoch = 1

    set_random_seed(args.seed)
    test(args)
