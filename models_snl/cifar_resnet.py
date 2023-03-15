from __future__ import absolute_import

'''
This file is from: https://raw.githubusercontent.com/bearpaw/pytorch-classification/master/models/cifar/resnet.py
by Wei Yang
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.nn.init as init
from .init_utils import weights_init


# __all__ = ['ResNet', 'resnet34_in', 'resnet50_in', 'resnet18_in', 'resnet9_in',
#            'wide_resnet22_8', 'wide_resnet_22_8_drop02', 'wide_resnet_28_10_drop02', 'wide_resnet_28_12_drop02', 'wide_resnet_16_8_drop02'
#            'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 
#            'vgg19_bn', 'vgg19', 'lenet_5_caffe']


def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

def drop_path_(x, drop_prob, training):
    if training and drop_prob > 0.:
        keep_prob = 1. - drop_prob
        # per data point mask; assuming x in cuda.
        mask = torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
        x.div_(keep_prob).mul_(mask)

    return x



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='B'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        # out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn1(self.conv1(x))
        out = self.relu(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class BasicBlock_IN(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride, feature_size):
        super(BasicBlock_IN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.relu(out)
        
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet_IN(nn.Module):
    def __init__(self, block, num_blocks, config):
        super(ResNet_IN, self).__init__()
        self.in_planes = 64
        if config.dataset in ['cifar10', 'cifar100']:
            self.feature_size = 32
            self.last_dim = 4
            print("CIFAR10/100 Setting")
        elif config.dataset in ['tiny_imagenet']:
            self.feature_size = 64
            self.last_dim = 8
            print("Tiny_ImageNet Setting")
            print("num_classes: ", config.num_classes)
        else:
            raise ValueError("Dataset not implemented for ResNet_IN")
        
        self.config = config
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=args.stride, padding=1, bias=False)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # self.alpha = LearnableAlpha(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, config.num_classes)

        self.apply(_weights_init)

        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            self.feature_size = self.feature_size // 2 if stride == 2 else self.feature_size
            layers.append(block(self.in_planes, planes, stride, self.feature_size))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, self.last_dim)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

def resnet9_in(config):
    return ResNet_IN(BasicBlock_IN, [1, 1, 1, 1], config=config)

def resnet18_in(config):
    return ResNet_IN(BasicBlock_IN, [2, 2, 2, 2], config=config)

def resnet34_in(config):
    return ResNet_IN(BasicBlock_IN, [3, 4, 6, 3], config=config)

def resnet50_in(config):
    return ResNet_IN(BasicBlock_IN, [3, 4, 14, 3], config=config)
    

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class WideBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, feature_size, dropRate=0.0):
        super(WideBasicBlock, self).__init__()
        self.equalInOut = (in_planes == out_planes)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU()

        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)

        self.droprate = dropRate
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu(self.bn1(x))
        else:
            out = self.relu(self.bn1(x))
        # out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        out = self.relu(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, feature_size, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, feature_size, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, feature_size, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, feature_size, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, config, depth=22, widen_factor=8, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        if config.dataset in ['cifar10', 'cifar100']:
            self.feature_size = 32
            self.last_dim = 8
        elif config.dataset in ['tiny_imagenet']:
            self.feature_size = 64
            self.last_dim = 16
        else:
            raise ValueError("Dataset not implemented in WideResNet")
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = WideBasicBlock
        self.config = config
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, self.feature_size, dropRate)
        # 2nd block
        self.feature_size = self.feature_size // 2
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, self.feature_size, dropRate)
        # 3rd block
        self.feature_size = self.feature_size // 2
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, self.feature_size, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU()
        # self.alpha = LearnableAlpha(nChannels[3])
        self.fc = nn.Linear(nChannels[3], config.num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.bn1(out)
        out = F.avg_pool2d(out, self.last_dim)
        out = self.fc(out.view(-1, self.nChannels))
        return out
    
def wide_resnet_16_8_drop02(config):
    return WideResNet(config=config, depth=16, widen_factor=8, dropRate=0.2)

def wide_resnet_22_8(config):
    return WideResNet(config=config, depth=22, widen_factor=8)

def wide_resnet_22_8_drop02(config):
    return WideResNet(config=config, depth=22, widen_factor=8, dropRate=0.2)

def wide_resnet_28_10_drop02(config):
    return WideResNet(config=config, depth=28, widen_factor=10, dropRate=0.2)

def wide_resnet_28_12_drop02(config):
    return WideResNet(config=config, depth=28, widen_factor=12, dropRate=0.2)

def wide_resnet(**kwargs):
    return WideResNet(**kwargs)


defaultcfg = {
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}


class VGG(nn.Module):
    def __init__(self, config, depth=19, init_weights=True, cfg=None, affine=True, batchnorm=True):
        super(VGG, self).__init__()
        if cfg is None:
            cfg = defaultcfg[depth]
        self._AFFINE = affine
        self.feature = self.make_layers(cfg, batchnorm)
        self.num_classes = config.num_classes
        
        self.classifier = nn.Linear(cfg[-1], config.num_classes)
        if init_weights:
            self.apply(weights_init)
        # if pretrained:
        #     model.load_state_dict(model_zoo.load_url(model_urls['vgg11_bn']))

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v, affine=self._AFFINE), nn.ReLU()]
                else:
                    layers += [conv2d, nn.ReLU()]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature(x)
        if self.num_classes == 200:
            x = nn.AvgPool2d(4)(x)
        else:
            x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        y = F.log_softmax(x, dim=1)
        return y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1.0)
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def vgg19(config):
    """VGG 19-layer model (configuration "E")"""
    return VGG(config)