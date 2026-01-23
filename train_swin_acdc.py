import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vision_transformer import SwinUnet as ViT_seg
from trainer_swin_acdc import trainer_acdc
from config import get_config

parser = argparse.ArgumentParser('Swin-Unet training for ACDC dataset')
parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file')
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='root dir for ACDC data')
parser.add_argument('--dataset', type=str,
                    default='ACDC', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=4, help='output channel of network (4 for ACDC)')
parser.add_argument('--output_dir', type=str, help='output dir')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.05,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--num_workers', type=int, default=4, help='number of workers for dataloader')
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, full: cache all data, part: sharding the dataset')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help='whether use gradient checkpointing to save memory')
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0 then no amp, if O1 then amp')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')

args = parser.parse_args()


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Load config
    config = get_config(args)
    
    # Create output directory
    if args.output_dir is None:
        args.output_dir = f"../model/swin_acdc_{args.img_size}_bs{args.batch_size}_lr{args.base_lr}_ep{args.max_epochs}"
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Create model
    model = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes)
    model = model.cuda()

    # Train
    trainer_acdc(args, model, args.output_dir)
