import torch
import torch.nn as nn
import os
import torch.nn.functional as F
import logging
from models_util import *
from models_cifar import *

def replace_relu(model, replacement_fn):
    for name, module in model.named_children():
        if isinstance(module, nn.ReLU):
            model._modules[name] = replacement_fn
        else:
            replace_relu(module, replacement_fn)

### Model with ReLU Replacement(RP)
class model_ReLU_RP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.model = model
        #### Initialize model architecture
        self.model = eval(config.arch + '()')
        self.Num_mask = config.Num_mask #### Initialize how many masks
        self.x_size = config.x_size #### Input image size, for example in cifar 10, it's [1, 3, 32, 32]
        self.sel_mask = 0
        ReLU_masked_model = eval(config.act_type + '()') #ReLU_masked()
        replace_relu(self, ReLU_masked_model)
        #### Get the name and model_stat of sparse ReLU model ####
        self._ReLU_sp_models = []
        for name, model_stat in self.named_modules(): 
            if 'ReLU_masked' in type(model_stat).__name__:
                self._ReLU_sp_models.append(model_stat)
        #### Get the name and model_stat of dropout model ####
        self.dropout_models = []
        for name, model_stat in self.named_modules(): 
            if 'Dropout2d' in type(model_stat).__name__:
                self.dropout_models.append(model_stat)
        #### Change number of masks to specified value ####
        for model_stat in self._ReLU_sp_models:
            model_stat.Num_mask = self.Num_mask

        #### Initialize alpha_aux pameters in ReLU_sp model ####
        #### through single step inference ####
        with torch.no_grad():
            in_mock_tensor = torch.Tensor(*self.x_size)
            self.forward(in_mock_tensor)
            del in_mock_tensor
            for model in self._ReLU_sp_models:
                model.init = 0

        ### Initialize _alpha_aux, _weights lists
        ### self._alpha_aux[i] is the ith _alpha_aux parameter
        self._alpha_aux = {}
        for i in range(self.Num_mask):
            self._alpha_aux[i] = []
        self._weights = []
        for name, parameter in self.named_parameters():
            if 'alpha_aux' in name:
                num = int(name.split("_")[-1])
                self._alpha_aux[num].append((name, parameter))                 
            else: 
                self._weights.append((name, parameter))
        
    def weights(self):
        for n, p in self._weights:
            yield p
    def named_weights(self):
        for n, p in self._weights:
            yield n, p
    def alpha_aux(self):
        for n, p in self._alpha_aux[self.sel_mask]:
            yield p
    def named_alpha_aux(self):
        for n, p in self._alpha_aux[self.sel_mask]:
            yield n, p
    ### Get Total number of gate parameter
    def _get_num_gates(self):
        with torch.no_grad():
            num_gates = torch.tensor(0.)
            for name, alpha_aux in self._alpha_aux[self.sel_mask]:
                num_gates += alpha_aux.numel()
        return num_gates
    def change_sel_mask(self, sel_mask = 0):
        self.sel_mask = sel_mask
        for model_stat in self._ReLU_sp_models:
            model_stat.sel_mask = sel_mask
    def change_dropout_ratio(self, dropout_ratio = 0):
        for model_stat in self.dropout_models:
            model_stat.p = dropout_ratio
    def train_fz_bn(self, freeze_bn=True, freeze_bn_affine=True, mode=True):
        """
            Override the default train() to freeze the BN parameters
        """
        # super(VGG, self).train(mode)
        self.train(mode)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                if freeze_bn_affine:
                    m.weight.requires_grad = not freeze_bn
                    m.bias.requires_grad = not freeze_bn

    def load_pretrained(self, pretrained_path = False):
        if pretrained_path:
            if os.path.isfile(pretrained_path):
                print("=> loading checkpoint '{}'".format(pretrained_path))
                checkpoint = torch.load(pretrained_path, map_location = "cpu")   
                # print('state_dict' in checkpoint.keys())  
                if 'state_dict' in checkpoint.keys():
                    pretrained_dict = checkpoint['state_dict']
                else:
                    pretrained_dict = checkpoint
                # pretrained_dict = checkpoint
                model_dict = self.state_dict()
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                
                # print(pretrained_dict)
                # print({k for k, v in pretrained_dict.items()})
                # print({k for k, v in model_dict.items()})
                # exit(0)
                model_dict.update(pretrained_dict) 
                self.load_state_dict(model_dict)
            else:
                print("=> no checkpoint found at '{}'".format(pretrained_path))

    def load_check_point(self, check_point_path = False):
        if os.path.isfile(check_point_path):
            print("=> loading checkpoint from '{}'".format(check_point_path))
            checkpoint = torch.load(check_point_path, map_location = "cpu")
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            self.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint at epoch {}"
                  .format(checkpoint['epoch']))
            print("Is best result?: ", checkpoint['is_best'])
            return start_epoch, best_prec1
        else:
            print("=> no checkpoint found at '{}'".format(check_point_path))
               
    def save_checkpoint(self, epoch, best_prec1, is_best, filename='checkpoint.pth.tar'):
        """
        Save the training model
        """
        state = {
                'epoch': epoch + 1,
                'state_dict': self.state_dict(),
                'best_prec1': best_prec1,
                'is_best': is_best
            }
        torch.save(state, filename)
    def update_sparse_list(self):
        # Update the sparse_list correspond to alpha_aux
        # Format: [layer name, Total original ReLU count, Pruned count, Pruned percentage]
        # Update the global_sparsity
        # Format: [Total original ReLU count, Pruned count, Pruned percentage]
        if(hasattr(self, "sparse_list")):
            del self.sparse_list
        if(hasattr(self, "global_sparsity")):
            del self.global_sparsity

        self.sparse_list = []
        total_count_global = 0
        sparsity_count_global = 0
        with torch.no_grad():
            for name, param in self._alpha_aux[self.sel_mask]:
                weight_mask = 1 - STEFunction.apply(param)
                total_count = weight_mask.numel()
                sparsity_count = torch.sum(weight_mask).item()
                sparsity_pert = sparsity_count/total_count
                self.sparse_list.append([name, total_count, sparsity_count, sparsity_pert])
                total_count_global += total_count
                sparsity_count_global += sparsity_count
        sparsity_pert_global = sparsity_count_global/total_count_global
        self.global_sparsity = [total_count_global, sparsity_count_global, sparsity_pert_global]
    def print_sparse_list(self, logger):
        # remove formats
        org_formatters = []
        for handler in logger.handlers:
            org_formatters.append(handler.formatter)
            handler.setFormatter(logging.Formatter("%(message)s"))

        # Logging alpha data: 
        logger.info("####### ReLU Sparsity #######")
        logger.info("# Layer wise neuron ReLU sparsity for the model")
        logger.info("# Format: [layer name, Total original ReLU count, Pruned count, Pruned percentage]")
        for sparse_list in self.sparse_list:
            logger.info(sparse_list)
        logger.info("# Global ReLU neuron sparsity for the model")
        logger.info("# Format: [Total original ReLU count, Pruned count, Pruned percentage]")
        logger.info(self.global_sparsity)
        logger.info("########## End ###########")
        # restore formats
        for handler, formatter in zip(logger.handlers, org_formatters):
            handler.setFormatter(formatter)
    def forward(self, x):
        out = self.model(x)
        return out