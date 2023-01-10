import numpy as np
import torch
import math
import torchvision
from torch import nn
from torch.nn import functional as F
from torchvision import datasets, transforms
import warnings
from functools import partial

from models_util import *
from models_cifar import *

model_name = 'ResNet18'
model = ResNet18()
# def get_my_code(base):

#     class model_build(base, model_util):
#         def __init__(self,):
#           pass
#     return model_build
# # model_build(eval(model_name), model_util)
# model_Masked = get_my_code(model)
config = 1
model_ReLU_RP(model, config)