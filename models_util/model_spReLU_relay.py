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
        return global_density, sparse_list, sparse_pert_list, total_mask

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
        # out = self.act(torch.mul(x, neuron_relu_mask)) + torch.mul(x, neuron_pass_mask)
        out = torch.mul(F.relu(x), neuron_relu_mask) + torch.mul(x, neuron_pass_mask)
        # out = self.dropout(out)
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
        return global_density, sparse_list, sparse_pert_list, total_mask
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
        # out = self.act(torch.mul(x, neuron_relu_mask)) + ReLU_Pruned.apply(torch.mul(x, neuron_pass_mask))
        out = torch.mul(F.relu(x), neuron_relu_mask) + torch.mul(ReLU_Pruned.apply(x), neuron_pass_mask)
        # out = self.dropout(out)
        # out_relu = F.relu(x)
        # if (self.training and self.p > 0):
        #     sel = float(random.uniform(0, 1) < self.p)
        #     out_final = sel * out_relu + (1 - sel) * out
        #     return out_final
        # else:
        #     return out
        return out

### ReLU with run time initialization method
# mask1: bitmap, 1 means has ReLU, 0 means direct pass.
# mask2: bitmap, 1 means direct pass, 0 means have ReLU
# a*mask2: passed element
# a*mask1: element need to be ReLU
# ReLU(a*mask1) + a*mask2
class ReLU_masked_autopoly_relay(nn.Module):
    def __init__(self, config, Num_mask = 1, dropRate=0):
        super().__init__()
        self.Num_mask = Num_mask
        self.num_feature = 0
        self.current_feature = 0
        self.sel_mask = 0
        self.init = 1
        self.threshold = config.threshold
        self.degree = config.degree
        self.freezeact = config.freezeact
        self.scale_x2 = config.scale_x2
        self.out_act_rep = eval("x{}act_auto".format(self.degree))
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
        
        ### Initialize the channel wise polynoimal activation parameter
        if len(size) == 3:
            para_size = [1, size[0], 1, 1]
        elif len(size) == 1:
            para_size = [1, size[0]]
        else:
            print("Size with {} not supported".format(size))
            exit()
        exec("self.poly_para_{} = []".format(self.num_feature))
        for i in range(self.degree + 1):
            # print("channel size:", channel_size)
            # print(nn.Parameter(torch.Tensor(*channel_size)))
            setattr(self, "poly_para_{}_{}".format(self.num_feature, i), nn.Parameter(torch.Tensor(*para_size)))
            if i == 0:
                nn.init.uniform_(getattr(self, "poly_para_{}_{}".format(self.num_feature, i)), a = 0, b = 0.0001)
            elif i == 1:
                nn.init.uniform_(getattr(self, "poly_para_{}_{}".format(self.num_feature, i)), a = 1, b = 1.0001)
            elif i == 2:
                nn.init.uniform_(getattr(self, "poly_para_{}_{}".format(self.num_feature, i)), a = 0, b = 0.0001)
            else:
                print("we currently don't have support for degree higher than 2")
                exit()
            setattr(eval("self.poly_para_{}_{}".format(self.num_feature, i)), 'requires_grad', (not self.freezeact)) 
            exec("self.poly_para_{}.append(self.poly_para_{}_{})".format(self.num_feature, self.num_feature, i))
            # print(eval("self.poly_para_{}_{}".format(self.num_feature, i)))
            # print("Create poly_para_{}_{} successfully".format(self.num_feature, i))
        # print("Shape: \n", eval("self.poly_para_{}[0].shape".format(self.num_feature)))
        # print("Value: \n", eval("self.poly_para_{}".format(self.num_feature)))
    
    ## aggregate the polynomial parameter to a list
    def expand_aggr_poly(self,):
        for current_feature in range(self.num_feature):
            exec("self.poly_para_{} = []".format(current_feature))
            for i in range(self.degree + 1):
                exec("self.poly_para_{}.append(self.poly_para_{}_{})".format(current_feature, current_feature, i))
     
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
        return global_density, sparse_list, sparse_pert_list, total_mask

    def forward(self, x):
        ### Initialize the parameter at the beginning
        if self.init:
            x_size = list(x.size())[1:] ### Ignore batch size dimension
            # var_map = x.var_map
            self.init_w_aux(x_size)
            neuron_relu_mask = STEFunction_relay.apply(getattr(self, "alpha_aux_{}_{}".format(self.num_feature, self.sel_mask)), 
                                                getattr(self, "alpha_mask_{}_{}".format(self.num_feature, self.sel_mask))) ### Mask for element which applies ReLU
            out_act_rep = self.out_act_rep
            if self.degree == 2:
                out_act_rep = partial(out_act_rep, para = eval("self.poly_para_{}".format(self.num_feature)), scale_x2 = self.scale_x2)
            else:
                out_act_rep = partial(out_act_rep, para = eval("self.poly_para_{}".format(self.num_feature)))
            self.num_feature += 1
        ### Conduct recurrently inference during normal inference and training
        else:
            if self.current_feature == 0:
                self.expand_aggr_poly()
                self.update_mask()

            # print("Current used: ", getattr(self, "alpha_aux_{}_{}".format(self.current_feature, self.sel_mask)))
            neuron_relu_mask = STEFunction_relay.apply(getattr(self, "alpha_aux_{}_{}".format(self.current_feature, self.sel_mask)), 
                                                getattr(self, "alpha_mask_{}_{}".format(self.current_feature, self.sel_mask))) ### Mask for element which applies ReLU
            # act_choice = eval(f"self.var_map_{self.current_feature}")
            # out_act_rep = eval("self.act_d{}_var{}".format(self.degree, act_choice))
            out_act_rep = self.out_act_rep
            if self.degree == 2:
                out_act_rep = partial(out_act_rep, para = eval("self.poly_para_{}".format(self.current_feature)), scale_x2 = self.scale_x2)
            else:
                out_act_rep = partial(out_act_rep, para = eval("self.poly_para_{}".format(self.current_feature)))

            self.current_feature = (self.current_feature + 1) % self.num_feature

        neuron_pass_mask = 1 - neuron_relu_mask  ### Mask for element which ignore ReLU
        # print(neuron_relu_mask)
        # print(neuron_pass_mask)


        # out2 = torch.mul(self.act(x), neuron_relu_mask.expand_as(x)) + torch.mul(out_act_rep(x), neuron_pass_mask.expand_as(x))
        # out = self.act(torch.mul(x, neuron_relu_mask.expand_as(x))) + torch.mul(out_act_rep(x), neuron_pass_mask.expand_as(x))

        out = torch.mul(F.relu(x), neuron_relu_mask.expand_as(x)) + torch.mul(out_act_rep(x), neuron_pass_mask.expand_as(x))
        # out = F.relu(torch.mul(x, neuron_relu_mask.expand_as(x))) + torch.mul(out_act_rep(x), neuron_pass_mask.expand_as(x))


        # diff = torch.norm(out - out2)
        # print("The difference: ", diff)

        # out = self.act(torch.mul(x, neuron_relu_mask)) + torch.mul(x, neuron_pass_mask)
        # out = torch.mul(self.act(x), neuron_relu_mask) + torch.mul(x, neuron_pass_mask)


        # if not self.training:
        #     with torch.no_grad():
        #         test_out = (self.act(x) - out).pow(2).sum().sqrt()
        #         print(test_out)


        # if not self.training:
        #     with torch.no_grad():
        #         neuron_relu_mask_test = neuron_relu_mask.sum()
        #         neuron_pass_mask_test = neuron_pass_mask.sum()
        #         print(out_act_rep)
        #         print(neuron_relu_mask_test, neuron_pass_mask_test)

        # out = self.dropout(out)
        
        # if (self.training and self.p > 0):
        #     out_relu = F.relu(x)
        #     sel = float(random.uniform(0, 1) < self.p)
        #     out_final = sel * out_relu + (1 - sel) * out
        #     return out_final
        # else:
        #     return out
        return out


### ReLU with run time initialization method
# mask1: bitmap, 1 means has ReLU, 0 means direct pass.
# mask2: bitmap, 1 means direct pass, 0 means have ReLU
# a*mask2: passed element
# a*mask1: element need to be ReLU
# ReLU(a*mask1) + a*mask2
class ReLU_masked_dapa_relay(nn.Module):
    def __init__(self, config, Num_mask = 1, dropRate=0):
        super().__init__()
        self.Num_mask = Num_mask
        self.num_feature = 0
        self.current_feature = 0
        self.sel_mask = 0
        self.init = 1
        self.threshold = config.threshold
        self.degree = config.degree
        self.freezeact = config.freezeact
        self.scale_x1 = config.scale_x1
        self.scale_x2 = config.scale_x2
        self.clip_x2 = config.clip_x2
        self.clip_x2_bool = config.clip_x2_bool
        self.out_act_rep = eval("x{}act_auto".format(self.degree))
        self.var_min = config.var_min
        self.dropout = nn.Dropout2d(p=dropRate, inplace=True)
        self.p = dropRate
    

        self.itr_cnt = nn.Parameter(torch.Tensor([0.]))
    @torch.no_grad()
    def init_w_aux(self, size):
        for i in range(self.Num_mask):
            setattr(self, "alpha_aux_{}_{}".format(self.num_feature, i), nn.Parameter(torch.Tensor(*size)))
            setattr(self, "alpha_mask_{}_{}".format(self.num_feature, i), nn.Parameter(torch.Tensor(*size)))
            nn.init.uniform_(getattr(self, "alpha_aux_{}_{}".format(self.num_feature, i)), a = 0, b = 1) # weight init for aux parameter, can be truncated normal
            nn.init.constant_(getattr(self, "alpha_mask_{}_{}".format(self.num_feature, i)), 1)
            setattr(eval('self.alpha_mask_{}_{}'.format(self.num_feature, i)), 'requires_grad', False)
        
        ### Initialize the channel wise polynoimal activation parameter
        if len(size) == 3:
            para_size = [1, size[0], 1, 1]
            #### Create the batch norm utility to get the running mean and variance of the activation function.
            setattr(self, "bn_{}".format(self.num_feature), nn.BatchNorm2d(num_features = size[0], affine=False))
        elif len(size) == 1:
            para_size = [1, size[0]]
            #### Create the batch norm utility to get the running mean and variance of the activation function.
            setattr(self, "bn_{}".format(self.num_feature), nn.BatchNorm1d(num_features = size[0], affine=False))
        else:
            print("Size with {} not supported".format(size))
            exit()
        exec("self.poly_para_{} = []".format(self.num_feature))
        for i in range(self.degree + 1):
            setattr(self, "poly_para_{}_{}".format(self.num_feature, i), nn.Parameter(torch.Tensor(*para_size)))
            if i == 0:
                nn.init.uniform_(getattr(self, "poly_para_{}_{}".format(self.num_feature, i)), a = 0, b = 0.0001)
            elif i == 1:
                nn.init.uniform_(getattr(self, "poly_para_{}_{}".format(self.num_feature, i)), a = 1, b = 1.0001)
            elif i == 2:
                nn.init.uniform_(getattr(self, "poly_para_{}_{}".format(self.num_feature, i)), a = 0, b = 0.0001)
            else:
                print("we currently don't have support for degree higher than 2")
                exit()
            #### Make the polynomial parameter to be trainable
            setattr(eval("self.poly_para_{}_{}".format(self.num_feature, i)), 'requires_grad', True) 
            exec("self.poly_para_{}.append(self.poly_para_{}_{})".format(self.num_feature, self.num_feature, i))

        
        
    ## aggregate the polynomial parameter to a list
    def expand_aggr_poly(self,):
        for current_feature in range(self.num_feature):
            exec("self.poly_para_{} = []".format(current_feature))
            for i in range(self.degree + 1):
                exec("self.poly_para_{}.append(self.poly_para_{}_{})".format(current_feature, current_feature, i))

    @torch.no_grad()
    def update_poly(self, x):
        #### Stop running mean and variance calculation after 508th iteration
        if self.current_feature == 0 and self.itr_cnt < 508:
            self.itr_cnt += 1
        bn_layer = eval("self.bn_{}".format(self.current_feature))
        bn_out = bn_layer(x)
        # print("Itr {}, Batch norm {} running mean:".format(self.itr_cnt, self.current_feature))
        # print(bn_layer.running_mean)
        # print("Itr {}, Batch norm {} running variance:".format(self.itr_cnt, self.current_feature))
        # print(bn_layer.running_var)
        if ((self.itr_cnt%10) == 0):
            u = bn_layer.running_mean
            # v = torch.clip(bn_layer.running_var, min=0.04)
            v = torch.clip(bn_layer.running_var, min=self.var_min)
            para = eval("approx_{}rd_torch(u, v)".format(self.degree))
            para = list(para)
            para[1] = para[1]/self.scale_x1
            # if self.degree == 2: 
            #     para[2] = para[2]/self.scale_x2
            

            # print("Itr {}, replacement parameter in layer {}:".format(self.itr_cnt, self.current_feature))
            # print("Bias: ", para[0])
            # print("Weight: ", para[1])
            # exit()/

            # Ignore batch size dimension
            size = list(x.size())[1:]
            if len(size) == 3:
                para_size = [1, size[0], 1, 1]
            elif len(size) == 1:
                para_size = [1, size[0]]
            else:
                print("Size with {} not supported".format(size))
                exit()
            for i in range(self.degree + 1):
                getattr(self, "poly_para_{}_{}".format(self.current_feature, i)).data.copy_(para[i].view(*para_size))
        # print("Updated successfully: {}", eval("self.poly_para_{}_{}".format(self.current_feature, i)))
        # exit()
        
        ############### Clip the polynomial parameter of x2 term to -0.2 to 0.2
        if self.clip_x2_bool:
            getattr(self, "poly_para_{}_{}".format(self.current_feature, 2)).data.copy_(
                        torch.clamp(eval("self.poly_para_{}_{}".format(self.current_feature, 2)), -1 * self.clip_x2, self.clip_x2))

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
        return global_density, sparse_list, sparse_pert_list, total_mask

    def forward(self, x):
        ### Initialize the parameter at the beginning
        if self.init:
            x_size = list(x.size())[1:] ### Ignore batch size dimension
            self.init_w_aux(x_size)
            neuron_relu_mask = STEFunction_relay.apply(getattr(self, "alpha_aux_{}_{}".format(self.num_feature, self.sel_mask)), 
                                                getattr(self, "alpha_mask_{}_{}".format(self.num_feature, self.sel_mask))) ### Mask for element which applies ReLU
            out_act_rep = self.out_act_rep
            if self.degree == 2:
                out_act_rep = partial(out_act_rep, para = eval("self.poly_para_{}".format(self.num_feature)), scale_x2 = self.scale_x2)
            else:
                out_act_rep = partial(out_act_rep, para = eval("self.poly_para_{}".format(self.num_feature)), scale_x1 = self.scale_x1)
            

            self.num_feature += 1
        ### Conduct recurrently inference during normal inference and training
        else:
            if self.current_feature == 0:
                self.expand_aggr_poly()
                self.update_mask()

            # print("Current used: ", getattr(self, "alpha_aux_{}_{}".format(self.current_feature, self.sel_mask)))
            neuron_relu_mask = STEFunction_relay.apply(getattr(self, "alpha_aux_{}_{}".format(self.current_feature, self.sel_mask)), 
                                                getattr(self, "alpha_mask_{}_{}".format(self.current_feature, self.sel_mask))) ### Mask for element which applies ReLU
            out_act_rep = self.out_act_rep
            if self.degree == 2:
                out_act_rep = partial(out_act_rep, para = eval("self.poly_para_{}".format(self.current_feature)), scale_x2 = self.scale_x2)
            else:
                out_act_rep = partial(out_act_rep, para = eval("self.poly_para_{}".format(self.current_feature)), scale_x1 = self.scale_x1)
            self.update_poly(x)
            self.current_feature = (self.current_feature + 1) % self.num_feature

        neuron_pass_mask = 1 - neuron_relu_mask  ### Mask for element which ignore ReLU

        out = torch.mul(F.relu(x), neuron_relu_mask.expand_as(x)) + torch.mul(out_act_rep(x), neuron_pass_mask.expand_as(x))
        # out = self.dropout(out)
        
        # if (self.training and self.p > 0):
        #     out_relu = F.relu(x)
        #     sel = float(random.uniform(0, 1) < self.p)
        #     out_final = sel * out_relu + (1 - sel) * out
        #     return out_final
        # else:
        #     return out
        return out