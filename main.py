import argparse

from train import *

## Parser 생성하기
parser = argparse.ArgumentParser()

parser.add_argument("--mode", default="train", choices=["train", "test"], type=str, dest="mode")
parser.add_argument("--train_continue", default="on", choices=["on", "off"], type=str, dest="train_continue")
parser.add_argument('--data_parallel', default=True,action='store_true')

parser.add_argument("--lr", default=2e-4, type=float, dest="lr")
parser.add_argument("--batch_size", default=16, type=int, dest="batch_size")
parser.add_argument("--num_epoch", default=1000000, type=int, dest="num_epoch")

#parser.add_argument("--data_dir", default="../../dataset/images/out_temp_image", type=str, dest="data_dir")
parser.add_argument("--data_dir", default="./images/json_image_train", type=str, dest="data_dir")
#parser.add_argument("--data_dir", default="./images/out_temp_image", type=str, dest="data_dir")
parser.add_argument("--ckpt_dir", default="./checkpoint", type=str, dest="ckpt_dir")
parser.add_argument("--log_dir", default="./log", type=str, dest="log_dir")
parser.add_argument("--result_dir", default="./result", type=str, dest="result_dir")

parser.add_argument("--task", default="PInet", choices=['PInet'], type=str, dest="task")

parser.add_argument("--ny", default=270, type=int, dest="ny")
parser.add_argument("--nx", default=480, type=int, dest="nx")
parser.add_argument("--nch", default=3, type=int, dest="nch")
parser.add_argument("--nker", default=64, type=int, dest="nker")

parser.add_argument("--network", default="PInet", choices=["PInet"], type=str, dest="network")

args = parser.parse_args()

if __name__ == "__main__":
    if args.mode == "train":
        train(args)
    elif args.mode == "test":
        test(args)