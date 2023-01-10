""" Config class for search/augment """
import argparse
import os
from functools import partial
import torch


def get_parser(name):
    """ make default formatted parser """
    parser = argparse.ArgumentParser(name, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # print default value always
    parser.add_argument = partial(parser.add_argument, help=' ')
    return parser


def parse_gpus(gpus):
    if gpus == 'all':
        return list(range(torch.cuda.device_count()))
    else:
        return [int(s) for s in gpus.split(',')]


class BaseConfig(argparse.Namespace):
    def print_params(self, prtf=print):
        prtf("")
        prtf("Parameters:")
        for attr, value in sorted(vars(self).items()):
            prtf("{}={}".format(attr.upper(), value))
        prtf("")

    def as_markdown(self):
        """ Return configs as markdown format """
        text = "|name|value|  \n|-|-|  \n"
        for attr, value in sorted(vars(self).items()):
            text += "|{}|{}|  \n".format(attr, value)

        return text

class TrainCifarConfig(BaseConfig):
    def build_parser(self):
        parser = get_parser("CIFAR-10 Training config")
        parser.add_argument('--dataset',  default='cifar10', help='cifar10 / cifar100')
        parser.add_argument('--data_path', default='./data/', help='CIFAR10 / MNIST / FashionMNIST')
        parser.add_argument('--act_type', type=str, default='nn.ReLU', choices = ['nn.ReLU', 'ReLU_masked'],
             help='Which non-lienar function to be used in the training')
        parser.add_argument('--batch_size', type=int, default=128, help='batch size')
        parser.add_argument('--w_lr', type=float, default=0.1, help='lr for weights')
        parser.add_argument('--w_lr_min', type=float, default=0.00001, help='minimum lr for weights')
        parser.add_argument('--w_momentum', type=float, default=0.9, help='momentum for weights')
        parser.add_argument('--w_weight_decay', type=float, default=5e-4, help='weight decay for weights')
        parser.add_argument('--w_grad_clip', type=float, default=5., help='gradient clipping for weights')
        parser.add_argument('--w_decay_epoch', type=int, default=20, help='lr decay for training')
        parser.add_argument('--alpha_lr', type=float, default=5e-4, help='lr for alpha')
        parser.add_argument('--lamda', type=float, default=1e0, help='penalty iterm for ReLU mask')
        parser.add_argument('--alpha_weight_decay', type=float, default=1e-3, help='weight decay for alpha')
        parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
        parser.add_argument('--Num_mask', type=int, default=1, help='Number of pruning mask during training')
        parser.add_argument('--gpus', default='4', help='gpu device ids separated by comma. `all` indicates use all gpus.')
        parser.add_argument('--epochs', type=int, default=200, help='# of training epochs')
        parser.add_argument('--mask_epochs', type=int, default=0, help='Training mask epochs')
        parser.add_argument('--seed', type=int, default=2, help='random seed')
        parser.add_argument('--workers', type=int, default=4, help='# of workers')
        parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
        # parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
        parser.add_argument('-e', '--evaluate', default=None, type=str, metavar='PATH',
                            help='path to checkpoint (default: none), evaluate model on validation set')
        parser.add_argument('--pretrained_path', help='Pretained model path')
        parser.add_argument('--checkpoint_path', help='Checkpoint path')
        # parser.add_argument('--pool_type', default='nn.AvgPool2d', help='Pooling layer type')
        # parser.add_argument('--pool_type', default='nn.MaxPool2d', help='Pooling layer type')
        # parser.add_argument('--arch', default='vgg16', help='Model architecture type')
        parser.add_argument('--arch', default='ResNet18', help='Model architecture type') #'resnet18'
        parser.add_argument('--dropout', type=float, default=0, help='Dropout ratio')
        parser.add_argument('--optim', type=str, default='cosine',choices = ['cosine', 'cosine_rst'], help='Optimizer choice')
        parser.add_argument('--precision', type=str, default='full', choices = ['full', 'half'], help='Full precision training or half precision training')
        parser.add_argument('--ext', default='baseline', help='Extension name')
        
        return parser

    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()
        super().__init__(**vars(args))

        # self.data_path = './data/'
        # self.path = os.path.join(f'searchs_{self.arch}', self.name + str("_{:.0e}".format(self.lat_lamda) + '_Finetune'))
        if self.evaluate:
            str_first = self.optim + '_' + ("lambda{}".format(self.lamda) if self.act_type != 'nn.ReLU' else 'baseline_')
            self.path = os.path.join("evaluate_cifar", f'{self.arch}_{self.dataset}', str_first + str("lr{}ep{}_{}".format(self.w_lr, self.epochs, self.ext)))
        else:
            str_first = self.optim + '_' + ("lambda{}".format(self.lamda) if self.act_type != 'nn.ReLU' else 'baseline_')
            self.path = os.path.join("train_cifar", f'{self.arch}_{self.dataset}', str_first + str("lr{}ep{}_{}".format(self.w_lr, self.epochs, self.ext)))
        # self.eval_log = os.path.join("evaluate", f'evaluate_{self.arch}', self.name + str("_lat_lmd_{:.0e}_lr{}ep{}".format(self.lat_lamda, self.w_lr, self.epochs)))
        self.plot_path = os.path.join(self.path, 'plots')
        self.gpus = parse_gpus(self.gpus)
        if self.dataset == "cifar10":
            self.num_classes = 10
            self.x_size = [1, 3, 32, 32]
        elif self.dataset == "cifar100":
            self.num_classes = 100
            self.x_size = [1, 3, 32, 32]

class ImageNetConfig(BaseConfig):
    def build_parser(self):
        parser = get_parser("Train ImageNet config")
        parser.add_argument('--name', default = "imagenet")
        parser.add_argument('--data', metavar='DIR', default='/data/imagenet/',
                            help='path to dataset (default: imagenet)')
        parser.add_argument('--arch', default='resnet18', help='Model architecture type')

        parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                            help='number of data loading workers (default: 4)')
        parser.add_argument('--epochs', default=50, type=int, metavar='N',
                            help='number of total epochs to run')
        parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                            help='manual epoch number (useful on restarts)')
        parser.add_argument('-b', '--batch-size', default=256, type=int,
                            metavar='N',
                            help='mini-batch size (default: 256), this is the total '
                                'batch size of all GPUs on the current node when '
                                'using Data Parallel or Distributed Data Parallel')
        parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                            metavar='LR', help='initial learning rate', dest='lr')
        parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                            help='momentum')
        parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                            metavar='W', help='weight decay (default: 1e-4)',
                            dest='weight_decay')
        parser.add_argument('-p', '--print-freq', default=100, type=int,
                            metavar='N', help='print frequency (default: 100)')
        parser.add_argument('--resume', default='', type=str, metavar='PATH',
                            help='path to latest checkpoint (default: none)')
        # parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
        #                     help='evaluate model on validation set')        
        parser.add_argument('-e', '--evaluate', default=None, type=str, metavar='PATH',
                            help='path to checkpoint (default: none)')
        parser.add_argument('--world-size', default=-1, type=int,
                            help='number of nodes for distributed training')
        parser.add_argument('--rank', default=-1, type=int,
                            help='node rank for distributed training')
        parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                            help='url used to set up distributed training')
        parser.add_argument('--dist-backend', default='nccl', type=str,
                            help='distributed backend')
        parser.add_argument('--seed', default=0, type=int,
                            help='seed for initializing training. ')
        parser.add_argument('--gpus', default='0', help='gpu device ids separated by comma. `all` indicates use all gpus.')

        parser.add_argument('--multiprocessing-distributed', action='store_true',
                            help='Use multi-processing distributed training to launch '
                                'N processes per node, which has N GPUs. This is the '
                                'fastest way to use PyTorch for either single node or '
                                'multi node data parallel training')
        
        parser.add_argument('--decay', type=int, default=30, help='Weight decay step size')
        parser.add_argument('--pretrained_NAS_path', help='Pretained NAS model path')
        parser.add_argument('--pretrained_path', help='Pretained model path')
        parser.add_argument('--all_poly_avgpl', default=True, help='Pretained NAS model path')
        parser.add_argument('--lat_lamda', type=float, default=0, help='Lamda for alpha latency loss constraint')
        parser.add_argument('--act_type', default='GateAct', help='Non-linear activation type')
        parser.add_argument('--pool_type', default='GatePool', help='Pooling layer type')
        parser.add_argument('--ext', default='', type=str,
                            help='self-defined extension for saved location')
        parser.add_argument('--act_rep_epoch', type=float, default = 1, 
                                help='Number of gradually replacement epochs for activation layer')
        parser.add_argument('--x2act_scale', type=float, default=1, help='scale of x2act scale*w*x2')

        return parser

    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()
        super().__init__(**vars(args))
        # self.path = os.path.join("finetune", f'finetune_{self.arch}', self.name + str("_lat_lmd_{:.0e}_lr{}ep{}".format(self.lat_lamda, self.w_lr, self.epochs)))
        
        self.path = os.path.join('finetune_imagenet', f'finetune_imagenet_{self.arch}', 
                                    str("_lat_lmd_{:.0e}_lr{}_ep{}_seed{}_{}".format(self.lat_lamda, self.lr, self.epochs, args.seed, args.ext)))
        # self.path = os.path.join('finetune_imagenet', f'finetune_imagenet_{self.arch}', 
        #                             f'imagenet_{self.lat_lamda}_Finetune_lr{self.lr}_dcy{self.decay}{args.ext}')
        self.plot_path = os.path.join(self.path, 'plots')
        self.gpus = parse_gpus(self.gpus)

