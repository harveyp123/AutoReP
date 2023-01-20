import numpy as np
import torch
import math
import torchvision
from torch import nn
from torch.nn import functional as F
from torchvision import datasets, transforms
import warnings
from functools import partial
import random
from models_util import *

class STEFunction_relay(torch.autograd.Function):
    """ define straight through estimator with overrided gradient (gate) """
    @staticmethod
    def forward(ctx, input, mask):
        ctx.save_for_backward(input)
        return mask

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return torch.mul(F.softplus(input), grad_output), None

### ReLU with run time initialization method
# mask1: bitmap, 1 means has ReLU, 0 means direct pass.
# mask2: bitmap, 1 means direct pass, 0 means have ReLU
# a*mask2: passed element
# a*mask1: element need to be ReLU
# ReLU(a*mask1) + a*mask2
class ReLU_masked_relay(nn.Module):
    def __init__(self, config, Num_mask = 1, dropRate=0):
        super().__init__()
        self.Num_mask = Num_mask
        self.num_feature = 0
        self.current_feature = 0
        self.sel_mask = 0
        self.init = 1
        self.threshold = config.threshold
        self.act = partial(F.relu, inplace = True)
        self.dropout = nn.Dropout2d(p=dropRate, inplace=True)
    @torch.no_grad()
    def init_w_aux(self, size):
        for i in range(self.Num_mask):
            setattr(self, "alpha_aux_{}_{}".format(self.num_feature, i), nn.Parameter(torch.Tensor(*size)))
            setattr(self, "alpha_mask_{}_{}".format(self.num_feature, i), nn.Parameter(torch.Tensor(*size)))
            nn.init.uniform_(getattr(self, "alpha_aux_{}_{}".format(self.num_feature, i)), a = 0, b = 1) # weight init for aux parameter, can be truncated normal
            nn.init.constant_(getattr(self, "alpha_mask_{}_{}".format(self.num_feature, i)), 1)
            setattr(eval('self.alpha_mask_{}_{}'.format(self.num_feature, i)), 'requires_grad', False)

    @torch.no_grad()
    def update_mask(self):
        for feature_i in range(self.num_feature):
            mask_old = getattr(self, "alpha_mask_{}_{}".format(feature_i, self.sel_mask)).data
            in_tensor = getattr(self, "alpha_aux_{}_{}".format(feature_i, self.sel_mask)).data
            mask_new = mask_old * (in_tensor > (-1) * self.threshold).float() + (1 - mask_old) * (in_tensor > self.threshold).float()
            getattr(self, "alpha_mask_{}_{}".format(feature_i, self.sel_mask)).data.copy_(mask_new)

    def mask_density_forward(self):
        l0_reg = 0
        sparse_list = []
        sparse_pert_list = []
        total_mask = 0
        for feature_i in range(self.num_feature):
            neuron_mask = STEFunction_relay.apply(getattr(self, "alpha_aux_{}_{}".format(feature_i, self.sel_mask)), 
                                            getattr(self, "alpha_mask_{}_{}".format(feature_i, self.sel_mask)))
            l0_reg += torch.sum(neuron_mask)
            sparse_list.append(torch.sum(neuron_mask).item())
            sparse_pert_list.append(sparse_list[-1]/neuron_mask.numel())
            total_mask += neuron_mask.numel()
        global_density = l0_reg/total_mask 
        return global_density, sparse_list, sparse_pert_list

    def forward(self, x):
        ### Initialize the parameter at the beginning
        if self.init:
            x_size = list(x.size())[1:] ### Ignore batch size dimension
            self.init_w_aux(x_size)
            neuron_relu_mask = STEFunction_relay.apply(getattr(self, "alpha_aux_{}_{}".format(self.num_feature, self.sel_mask)), 
                                                getattr(self, "alpha_mask_{}_{}".format(self.num_feature, self.sel_mask))) ### Mask for element which applies ReLU
            self.num_feature += 1
        ### Conduct recurrently inference during normal inference and training
        else:
            if self.current_feature == 0:
                self.update_mask()
            # print("Current used: ", getattr(self, "alpha_aux_{}_{}".format(self.current_feature, self.sel_mask)))
            neuron_relu_mask = STEFunction_relay.apply(getattr(self, "alpha_aux_{}_{}".format(self.current_feature, self.sel_mask)), 
                                                getattr(self, "alpha_mask_{}_{}".format(self.current_feature, self.sel_mask))) ### Mask for element which applies ReLU
            self.current_feature = (self.current_feature + 1) % self.num_feature
        neuron_pass_mask = 1 - neuron_relu_mask  ### Mask for element which ignore ReLU
        out = self.act(torch.mul(x, neuron_relu_mask)) + torch.mul(x, neuron_pass_mask)
        out = self.dropout(out)
        return out

### ReLU with run time initialization method
# mask1: bitmap, 1 means has ReLU, 0 means direct pass.
# mask2: bitmap, 1 means direct pass, 0 means have ReLU
# a*mask2: passed element
# a*mask1: element need to be ReLU
# ReLU(a*mask1) + a*mask2
class ReLU_masked_spgrad_relay(nn.Module):
    def __init__(self, config, Num_mask = 1, dropRate=0):
        super().__init__()
        self.Num_mask = Num_mask
        self.num_feature = 0
        self.current_feature = 0
        self.sel_mask = 0
        self.init = 1
        self.threshold = config.threshold
        self.act = partial(F.relu, inplace = True)
        self.dropout = nn.Dropout2d(p=dropRate, inplace=True)
        self.p = dropRate
    def init_w_aux(self, size):
        for i in range(self.Num_mask):
            setattr(self, "alpha_aux_{}_{}".format(self.num_feature, i), nn.Parameter(torch.Tensor(*size)))
            setattr(self, "alpha_mask_{}_{}".format(self.num_feature, i), nn.Parameter(torch.Tensor(*size)))
            nn.init.uniform_(getattr(self, "alpha_aux_{}_{}".format(self.num_feature, i)), a = 0, b = 1) # weight init for aux parameter, can be truncated normal
            nn.init.constant_(getattr(self, "alpha_mask_{}_{}".format(self.num_feature, i)), 1)
            setattr(eval('self.alpha_mask_{}_{}'.format(self.num_feature, i)), 'requires_grad', False)

    @torch.no_grad()
    def update_mask(self):
        for feature_i in range(self.num_feature):
            mask_old = getattr(self, "alpha_mask_{}_{}".format(feature_i, self.sel_mask)).data
            in_tensor = getattr(self, "alpha_aux_{}_{}".format(feature_i, self.sel_mask)).data
            mask_new = mask_old * (in_tensor > (-1) * self.threshold).float() + (1 - mask_old) * (in_tensor > self.threshold).float()
            getattr(self, "alpha_mask_{}_{}".format(feature_i, self.sel_mask)).data.copy_(mask_new)

    def mask_density_forward(self):
        l0_reg = 0
        sparse_list = []
        sparse_pert_list = []
        total_mask = 0
        for feature_i in range(self.num_feature):
            neuron_mask = STEFunction_relay.apply(getattr(self, "alpha_aux_{}_{}".format(feature_i, self.sel_mask)), 
                                            getattr(self, "alpha_mask_{}_{}".format(feature_i, self.sel_mask)))
            l0_reg += torch.sum(neuron_mask)
            sparse_list.append(torch.sum(neuron_mask).item())
            sparse_pert_list.append(sparse_list[-1]/neuron_mask.numel())
            total_mask += neuron_mask.numel()
        global_density = l0_reg/total_mask 
        return global_density, sparse_list, sparse_pert_list
    def forward(self, x):
        ### Initialize the parameter at the beginning
        if self.init:
            x_size = list(x.size())[1:] ### Ignore batch size dimension
            self.init_w_aux(x_size)
            neuron_relu_mask = STEFunction_relay.apply(getattr(self, "alpha_aux_{}_{}".format(self.num_feature, self.sel_mask)), 
                                                getattr(self, "alpha_mask_{}_{}".format(self.num_feature, self.sel_mask))) ### Mask for element which applies ReLU
            self.num_feature += 1
        ### Conduct recurrently inference during normal inference and training
        else:
            # print("Current used: ", getattr(self, "alpha_aux_{}_{}".format(self.current_feature, self.sel_mask)))
            if self.current_feature == 0:
                # print("Update mask!!!!!!")
                self.update_mask()
            neuron_relu_mask = STEFunction_relay.apply(getattr(self, "alpha_aux_{}_{}".format(self.current_feature, self.sel_mask)), 
                                                getattr(self, "alpha_mask_{}_{}".format(self.current_feature, self.sel_mask))) ### Mask for element which applies ReLU
            self.current_feature = (self.current_feature + 1) % self.num_feature
        neuron_pass_mask = 1 - neuron_relu_mask  ### Mask for element which ignore ReLU
        out = self.act(torch.mul(x, neuron_relu_mask)) + ReLU_Pruned.apply(torch.mul(x, neuron_pass_mask))
        out = self.dropout(out)
        out_relu = F.relu(x)
        if (self.training and self.p > 0):
            sel = float(random.uniform(0, 1) < self.p)
            out_final = sel * out_relu + (1 - sel) * out
            return out_final
        else:
            return out

### ReLU with run time initialization method
# mask1: bitmap, 1 means has ReLU, 0 means direct pass.
# mask2: bitmap, 1 means direct pass, 0 means have ReLU
# a*mask2: passed element
# a*mask1: element need to be ReLU
# ReLU(a*mask1) + a*mask2
class ReLU_masked_poly_relay(nn.Module):
    def __init__(self, config, Num_mask = 1, dropRate=0):
        super().__init__()
        self.Num_mask = Num_mask
        self.num_feature = 0
        self.current_feature = 0
        self.sel_mask = 0
        self.init = 1
        self.threshold = config.threshold
        self.act = partial(F.relu, inplace = True)
        self.act2 = partial(x2act, scale_x2 = config.scale_x2, scale_x = config.scale_x)
        self.dropout = nn.Dropout2d(p=dropRate, inplace=True)
        self.p = dropRate
    @torch.no_grad()
    def init_w_aux(self, size):
        for i in range(self.Num_mask):
            setattr(self, "alpha_aux_{}_{}".format(self.num_feature, i), nn.Parameter(torch.Tensor(*size)))
            setattr(self, "alpha_mask_{}_{}".format(self.num_feature, i), nn.Parameter(torch.Tensor(*size)))
            nn.init.uniform_(getattr(self, "alpha_aux_{}_{}".format(self.num_feature, i)), a = 0, b = 1) # weight init for aux parameter, can be truncated normal
            nn.init.constant_(getattr(self, "alpha_mask_{}_{}".format(self.num_feature, i)), 1)
            setattr(eval('self.alpha_mask_{}_{}'.format(self.num_feature, i)), 'requires_grad', False)

    @torch.no_grad()
    def update_mask(self):
        for feature_i in range(self.num_feature):
            mask_old = getattr(self, "alpha_mask_{}_{}".format(feature_i, self.sel_mask)).data
            in_tensor = getattr(self, "alpha_aux_{}_{}".format(feature_i, self.sel_mask)).data
            mask_new = mask_old * (in_tensor > (-1) * self.threshold).float() + (1 - mask_old) * (in_tensor > self.threshold).float()
            getattr(self, "alpha_mask_{}_{}".format(feature_i, self.sel_mask)).data.copy_(mask_new)

    def mask_density_forward(self):
        l0_reg = 0
        sparse_list = []
        sparse_pert_list = []
        total_mask = 0
        for feature_i in range(self.num_feature):
            neuron_mask = STEFunction_relay.apply(getattr(self, "alpha_aux_{}_{}".format(feature_i, self.sel_mask)), 
                                            getattr(self, "alpha_mask_{}_{}".format(feature_i, self.sel_mask)))
            l0_reg += torch.sum(neuron_mask)
            sparse_list.append(torch.sum(neuron_mask).item())
            sparse_pert_list.append(sparse_list[-1]/neuron_mask.numel())
            total_mask += neuron_mask.numel()
        global_density = l0_reg/total_mask 
        return global_density, sparse_list, sparse_pert_list

    def forward(self, x):
        ### Initialize the parameter at the beginning
        if self.init:
            x_size = list(x.size())[1:] ### Ignore batch size dimension
            self.init_w_aux(x_size)
            neuron_relu_mask = STEFunction_relay.apply(getattr(self, "alpha_aux_{}_{}".format(self.num_feature, self.sel_mask)), 
                                                getattr(self, "alpha_mask_{}_{}".format(self.num_feature, self.sel_mask))) ### Mask for element which applies ReLU
            self.num_feature += 1
        ### Conduct recurrently inference during normal inference and training
        else:
            if self.current_feature == 0:
                self.update_mask()
            # print("Current used: ", getattr(self, "alpha_aux_{}_{}".format(self.current_feature, self.sel_mask)))
            neuron_relu_mask = STEFunction_relay.apply(getattr(self, "alpha_aux_{}_{}".format(self.current_feature, self.sel_mask)), 
                                                getattr(self, "alpha_mask_{}_{}".format(self.current_feature, self.sel_mask))) ### Mask for element which applies ReLU
            self.current_feature = (self.current_feature + 1) % self.num_feature
        neuron_pass_mask = 1 - neuron_relu_mask  ### Mask for element which ignore ReLU
        out = torch.mul(self.act(x), neuron_relu_mask) + torch.mul(self.act2(x), neuron_pass_mask)
        out = self.dropout(out)
        
        if (self.training and self.p > 0):
            out_relu = F.relu(x)
            sel = float(random.uniform(0, 1) < self.p)
            out_final = sel * out_relu + (1 - sel) * out
            return out_final
        else:
            return out