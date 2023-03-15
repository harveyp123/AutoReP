import torch
import torch.nn as nn
import os
import torch.nn.functional as F
import torch.nn.init as init

import logging
from models_util import *
from models_cifar import *
from models_snl import *
import torchvision

def replace_relu(model, replacement_fn):
    for name, module in model.named_children():
        if isinstance(module, nn.ReLU):
            model._modules[name] = replacement_fn
        else:
            replace_relu(module, replacement_fn)

def replace_siLU(model, replacement_fn):
    for name, module in model.named_children():
        if isinstance(module, nn.SiLU):
            model._modules[name] = replacement_fn
        else:
            replace_siLU(module, replacement_fn)     
# def replace_siLU(model):
#     for name, module in model.named_children():
#         if isinstance(module, nn.SiLU):
#             model._modules[name] = nn.ReLU
#         else:
#             replace_siLU(module)  

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)
class SoftTarget(nn.Module):
	'''
	Distilling the Knowledge in a Neural Network
	https://arxiv.org/pdf/1503.02531.pdf
	'''
	def __init__(self, T):
		super(SoftTarget, self).__init__()
		self.T = T

	def forward(self, out_s, out_t):
		loss = F.kl_div(F.log_softmax(out_s/self.T, dim=1),
						F.softmax(out_t/self.T, dim=1),
						reduction='batchmean') * self.T * self.T

		return loss
        
### Model with ReLU Replacement(RP)
class model_ReLU_RP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.model = model
        #### Initialize model architecture
        self.config = config
        self.arch = config.arch
        # if config.dataset == "cifar100":
        #     self.model = eval(config.arch + '(num_classes = 100)')
        # elif config.dataset == "cifar10":
        #     self.model = eval(config.arch + '(num_classes = 10)')
        # else:
        #     print("dataset not supported yet")

        if config.dataset != "imagenet":
            self.model = eval(config.arch + '(config)')
            self.model.apply(_weights_init)
        else:
            weight = ''
            if config.pretrained:
                if config.arch == 'resnet18':
                    weight = "weights = torchvision.models.ResNet18_Weights.DEFAULT"
                elif config.arch == 'resnet50':
                    weight = "weights = torchvision.models.ResNet50_Weights.DEFAULT"
                elif config.arch == "efficientnet_b0":
                    weight = "weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT"
                elif config.arch == "efficientnet_b2":#torchvision.models.efficientnet_b2
                    weight = "weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT"
                elif config.arch == "regnet_x_1_6gf":
                    weight = "weights = torchvision.models.RegNet_X_1_6GF_Weights.DEFAULT"
                elif config.arch == "regnet_x_800mf":
                    weight = "weights = torchvision.models.RegNet_X_800MF_Weights.DEFAULT"
            self.model = eval("torchvision.models." + config.arch + "({})".format(weight)) 
            # self.model = torchvision.models.regnet_x_1_6gf
            # self.model = eval("torchvision.models." + config.arch + "(pretrained = config.pretrained)") 
            #### Change maxpool to avepool
            if config.arch == 'resnet18':
                self.model.maxpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.Num_mask = config.Num_mask #### Initialize how many masks
        self.x_size = config.x_size #### Input image size, for example in cifar 10, it's [1, 3, 32, 32]
        self.sel_mask = 0
        ReLU_masked_model = eval(config.act_type + '(config)') #ReLU_masked()
        if "efficientnet" in config.arch:
            replace_siLU(self, ReLU_masked_model)
        else: 
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
        self.eval()
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
        self._weights_and_alpha = []
        for name, parameter in self.named_parameters():
            if 'alpha_aux' in name:
                num = int(name.split("_")[-1])
                self._alpha_aux[num].append((name, parameter))    
                self._weights_and_alpha.append((name, parameter))             
            else: 
                self._weights.append((name, parameter))
                self._weights_and_alpha.append((name, parameter))
            
        self._alpha_mask = {}
        for i in range(self.Num_mask):
            self._alpha_mask[i] = []
        for name, parameter in self.named_parameters():
            if 'alpha_mask' in name:
                num = int(name.split("_")[-1])
                self._alpha_mask[num].append((name, parameter))                 
    def weights(self):
        for n, p in self._weights:
            yield p
    def named_weights(self):
        for n, p in self._weights:
            yield n, p
    def weights_and_alpha(self):
        for n, p in self._weights_and_alpha:
            yield p
    def named_weights_and_alpha(self):
        for n, p in self._weights_and_alpha:
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
    def change_mask_dropout_ratio(self, dropout_ratio = 0):
        for model_stat in self._ReLU_sp_models:
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
                if (freeze_bn_affine and m.affine == True):
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
            if 'relay' in self.config.act_type:
                for name, param in self._alpha_mask[self.sel_mask]:
                    weight_mask = param.data
                    total_count = weight_mask.numel()
                    sparsity_count = torch.sum(weight_mask).item()
                    sparsity_pert = sparsity_count/total_count
                    self.sparse_list.append([name, total_count, sparsity_count, sparsity_pert])
                    total_count_global += total_count
                    sparsity_count_global += sparsity_count
                pass
            else:
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
        logger.info("# Format: [layer name, Total original ReLU count, remained count, remained percentage]")
        for sparse_list in self.sparse_list:
            logger.info(sparse_list)
        logger.info("# Global ReLU neuron sparsity for the model")
        logger.info("# Format: [Total original ReLU count, remained count, remained percentage]")
        logger.info(self.global_sparsity)
        logger.info("########## End ###########")
        if 'autopoly' in self.config.act_type:
            for model_stat in self._ReLU_sp_models:
                for current_feature in range(model_stat.num_feature):
                    messege = "\n Layer {} activation function parameter: ".format(current_feature)
                    logger.info(messege)
                    for i in range(self.config.degree + 1):
                        j = self.config.degree - i
                        logger.info("degree {} activation function parameter model_stat.poly_para_{}[{}]: ".format(j, current_feature, j))
                        logger.info(eval("torch.flatten(model_stat.poly_para_{}[{}])".format(current_feature, j)))
                    
        # restore formats
        for handler, formatter in zip(logger.handlers, org_formatters):
            handler.setFormatter(formatter)
    def forward(self, x):
        out = self.model(x)
        return out