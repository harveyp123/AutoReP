""" Search cell """
import os
import torch
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
from util_func.config import TrainCifarConfig
import util_func.utils as utils
# from models.search_cnn import SearchCNNController
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim
import torch.utils.data
import pytorch_warmup as warmup
from models_util import *
from models_cifar import *
config = TrainCifarConfig()

device = torch.device("cuda")

# tensorboard
writer = SummaryWriter(log_dir=os.path.join(config.path, "tb"))
writer.add_text('config', config.as_markdown(), 0)

logger = utils.get_logger(os.path.join(config.path, "{}.log".format(config.dataset)))
config.print_params(logger.info)


def main():
    logger.info("Logger is set - training start")

    # set default gpu device id
    torch.cuda.set_device(config.gpus[0])
    device = torch.device("cuda")
    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    torch.backends.cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss().to(device)

    model = model_ReLU_RP(config)
    model.criterion = criterion
    # if 'vgg' in config.arch:
    #     model = vgg.__dict__[config.arch](config, criterion, config.act_type, config.pool_type)
    # elif 'resnet' in config.arch:
    #     model = resnet.__dict__[config.arch](config, criterion, config.act_type, config.pool_type, num_classes = config.num_classes)
    # elif config.arch == "mobilenet_v2":
    #     model = mbnet.__dict__[config.arch](config, criterion, config.act_type, num_classes = config.num_classes)
    

    # config.pretrained_path = '/data3/hop20001/mpc_proj/PASNET_cifar10_search/pretrained/cifar10_baseline/vgg_checkpoint_best.tar'
    # config.pretrained_path = '/data3/hop20001/mpc_proj/PASNET_cifar10_search/pretrained/cifar10_baseline/resnet18.pt'
    ### finetune from checkpoint
    if config.pretrained_path:
        print("==> Load pretrained")
        model.load_pretrained(pretrained_path = config.pretrained_path)
    
    if(config.checkpoint_path):
        config.start_epoch, best_top1 = model.load_check_point(check_point_path = config.checkpoint_path)
    
    model = model.to(device)

    if config.dataset == "cifar10":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root=config.data_path, train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True),
            batch_size=config.batch_size, shuffle=True,
            num_workers=config.workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root=config.data_path, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=config.batch_size, shuffle=False,
            num_workers=config.workers, pin_memory=True)

    elif config.dataset == "cifar100":
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2023, 0.1994, 0.2010])
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(root=config.data_path, train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True),
            batch_size=config.batch_size, shuffle=True,
            num_workers=config.workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(root=config.data_path, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=config.batch_size, shuffle=False,
            num_workers=config.workers, pin_memory=True)

    if config.evaluate:

        # # ----------------------------
        checkpoint = torch.load(config.evaluate, map_location="cpu")
        model.load_state_dict(checkpoint['state_dict'])
        # print(model)
        from torchinfo import summary
        device = "cuda"
        summary(model, input_size=(1, 3, 32, 32), device=device, depth=3, verbose=2,
            col_names=["input_size",
                        "output_size",
                        "kernel_size"],
        )
        # model.print_alphas(logger)

        validate(val_loader, model, 0, len(val_loader))
        return
        # # ----------------------------
    # weights optimizer
    w_optim = torch.optim.SGD(model.weights(), config.w_lr, momentum=config.w_momentum,
                              weight_decay=config.w_weight_decay)
    # alphas optimizer
    if config.act_type != 'nn.ReLU':
        alpha_optim = torch.optim.Adam(model.alpha_aux(), config.alpha_lr, betas=(0.5, 0.999),
                                    weight_decay=config.alpha_weight_decay)
    if config.optim == 'cosine_rst':
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(w_optim, 1, T_mult=2, eta_min=config.w_lr_min) #, last_epoch = config.epochs
        T_mult = (config.epochs - config.mask_epochs) // config.mask_epochs
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(w_optim, config.mask_epochs, T_mult=T_mult, eta_min=config.w_lr_min) #, last_epoch = config.epochs
    else:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            w_optim, config.epochs, eta_min=config.w_lr_min)
    # warmup_scheduler = warmup.UntunedLinearWarmup(alpha_optim)

    #### Freeze batch normalization ####
    model.train_fz_bn(freeze_bn=True)
    # lambda_l1 = 1e-6
    # lambda_l2 = 5e-4
    lambda0 = config.lamda
    # training loop
    best_top1 = 0.
    for epoch in range(config.start_epoch, config.epochs):
        if config.act_type != 'nn.ReLU':
            model.update_sparse_list()
            model.print_sparse_list(logger)

        # training with mask update
        if epoch < config.mask_epochs:
            if config.precision == 'full':
                top1_train = train_mask(train_loader, model, w_optim, alpha_optim, lambda0, epoch)
            else:
                top1_train = train_mask_fp16(train_loader, model, w_optim, alpha_optim, lambda0, epoch)
        # training without mask update
        else:
            model.train_fz_bn(freeze_bn=False)
            model.change_dropout_ratio(config.dropout)
            if config.precision == 'full':
                top1_train = train(train_loader, model, w_optim, epoch)
            else:
                top1_train = train_fp16(train_loader, model, w_optim, epoch)
        # adjust_learning_rate(w_optim, epoch)
        lr_scheduler.step()
        # validation
        cur_step = (epoch+1) * len(train_loader)
        top1 = validate(val_loader, model, epoch, cur_step)

        # save 
        if epoch > config.mask_epochs:
            if best_top1 < top1:
                best_top1 = top1
                # best_genotype = genotype
                is_best = True
            else:
                is_best = False
            # utils.save_checkpoint(model, config.path, is_best)
            if is_best:
                save_path = os.path.join(config.path, 'best.pth.tar')
            else:
                save_path = os.path.join(config.path, 'checkpoint.pth.tar')
            model.save_checkpoint(epoch, best_top1, is_best, filename=save_path)
            logger.info("Current best Prec@1 = {:.4%}".format(best_top1))

        if (epoch % 20) == 0:
            logger.info("Perform validation on training dataset. ")
            top1_train_wo_dropout = validate_train(train_loader, model, epoch, cur_step)
            logger.info("Final train Prec@1 = {:.4%}".format(top1_train_wo_dropout))

    logger.info("Final best validation Prec@1 = {:.4%}".format(best_top1))
    # logger.info("Best Genotype = {}".format(best_genotype))

def train_mask(train_loader, model, w_optim, alpha_optim, lambda0, epoch):
    """
        Run one train epoch
    """
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()
    cur_step = epoch*len(train_loader)
    
    # switch to train mode
    total_mask = model._get_num_gates().item()
    model.train()

    for step, (input, target) in enumerate(train_loader):
        N = input.size(0)
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        alpha_optim.zero_grad()
        ### Compute cross entropy loss: ###
        output = model(input)
        ce_loss = model.criterion(output, target)

        sparse_list = []
        sparse_pert_list = []
        l0_reg = 0
        for name, param in model._alpha_aux[0]:
            neuron_mask = STEFunction.apply(param)
            l0_reg += torch.sum(neuron_mask)
            sparse_list.append(torch.sum(neuron_mask).item())
            sparse_pert_list.append(sparse_list[-1]/neuron_mask.numel())
        global_density = l0_reg/total_mask
        # compute gradient and do SGD step
        loss = ce_loss + lambda0*(global_density)
        loss.backward()
        alpha_optim.step()


        w_optim.zero_grad()
        output = model(input)
        ce_loss = model.criterion(output, target)
        ce_loss.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(model.weights(), config.w_grad_clip)
        w_optim.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1, prec5 = utils.accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), N)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)

        if step % config.print_freq == 0 or step == len(train_loader)-1:
            logger.info(
                "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                    epoch+1, config.epochs, step, len(train_loader)-1, losses=losses,
                    top1=top1, top5=top5))
            logger.info("layerwise density: " + str(sparse_list) + "\nlayerwise density percentage: "
                        + str([ '%.3f' % elem for elem in sparse_pert_list]) + "\nGlobal density: "
                        + str(global_density.item()))

        writer.add_scalar('train/loss', loss.item(), cur_step)
        writer.add_scalar('train/top1', prec1.item(), cur_step)
        writer.add_scalar('train/top5', prec5.item(), cur_step)
        cur_step += 1

    logger.info("Train: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1.avg))
    return top1.avg

def train_mask_fp16(train_loader, model, w_optim, alpha_optim, lambda0, epoch):
    """
        Run one train epoch
    """
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()
    cur_step = epoch*len(train_loader)
    
    # switch to train mode
    total_mask = model._get_num_gates().item()
    model.train()
    use_amp = True
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    for step, (input, target) in enumerate(train_loader):
        N = input.size(0)
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        alpha_optim.zero_grad()
        ### Compute cross entropy loss: ###
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            output = model(input)
            ce_loss = model.criterion(output, target)

        sparse_list = []
        sparse_pert_list = []
        l0_reg = 0
        for name, param in model._alpha_aux[0]:
            neuron_mask = STEFunction.apply(param)
            l0_reg += torch.sum(neuron_mask)
            sparse_list.append(torch.sum(neuron_mask).item())
            sparse_pert_list.append(sparse_list[-1]/neuron_mask.numel())
        global_sparsity = l0_reg/total_mask
        # compute gradient and do SGD step
        loss = ce_loss + lambda0*(global_sparsity)
        # loss.backward()
        # alpha_optim.step()
        scaler.scale(loss).backward()
        scaler.step(alpha_optim)
        scaler.update()

        w_optim.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            output = model(input)
            ce_loss = model.criterion(output, target)
        # ce_loss.backward()
        scaler.scale(ce_loss).backward()
        scaler.unscale_(w_optim)
        # gradient clipping
        nn.utils.clip_grad_norm_(model.weights(), config.w_grad_clip)
        # w_optim.step()
        scaler.step(w_optim)
        scaler.update()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1, prec5 = utils.accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), N)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)

        if step % config.print_freq == 0 or step == len(train_loader)-1:
            logger.info(
                "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                    epoch+1, config.epochs, step, len(train_loader)-1, losses=losses,
                    top1=top1, top5=top5))
            logger.info("layerwise density: " + str(sparse_list) + "\nlayerwise density percentage: "
                        + str([ '%.3f' % elem for elem in sparse_pert_list]) + "\nGlobal density: "
                        + str(global_sparsity.item()))

        writer.add_scalar('train/loss', loss.item(), cur_step)
        writer.add_scalar('train/top1', prec1.item(), cur_step)
        writer.add_scalar('train/top5', prec5.item(), cur_step)
        cur_step += 1

    logger.info("Train: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1.avg))
    return top1.avg

def train(train_loader, model, w_optim, epoch):
    """
        Run one train epoch
    """
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()
    cur_step = epoch*len(train_loader)
    
    # switch to train mode
    total_mask = model._get_num_gates().item()
    model.train()
    for step, (input, target) in enumerate(train_loader):
        N = input.size(0)
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        w_optim.zero_grad()
        output = model(input)
        loss = model.criterion(output, target)
        loss.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(model.weights(), config.w_grad_clip)
        w_optim.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1, prec5 = utils.accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), N)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)

        if step % config.print_freq == 0 or step == len(train_loader)-1:
            logger.info(
                "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                    epoch+1, config.epochs, step, len(train_loader)-1, losses=losses,
                    top1=top1, top5=top5))
        writer.add_scalar('train/loss', loss.item(), cur_step)
        writer.add_scalar('train/top1', prec1.item(), cur_step)
        writer.add_scalar('train/top5', prec5.item(), cur_step)
        cur_step += 1

    logger.info("Train: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1.avg))
    return top1.avg         

def train_fp16(train_loader, model, w_optim, epoch):
    """
        Run one train epoch
    """
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()
    cur_step = epoch*len(train_loader)
    
    # switch to train mode
    total_mask = model._get_num_gates().item()
    model.train()
    use_amp = True
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    for step, (input, target) in enumerate(train_loader):
        N = input.size(0)
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        # w_optim.zero_grad()
        # output = model(input)
        # loss = model.criterion(output, target)
        # loss.backward()
        # # gradient clipping
        # nn.utils.clip_grad_norm_(model.weights(), config.w_grad_clip)
        # w_optim.step()
        w_optim.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            output = model(input)
            loss = model.criterion(output, target)
        # ce_loss.backward()
        scaler.scale(loss).backward()
        scaler.unscale_(w_optim)
        # gradient clipping
        nn.utils.clip_grad_norm_(model.weights(), config.w_grad_clip)
        # w_optim.step()
        scaler.step(w_optim)
        scaler.update()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1, prec5 = utils.accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), N)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)

        if step % config.print_freq == 0 or step == len(train_loader)-1:
            logger.info(
                "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                    epoch+1, config.epochs, step, len(train_loader)-1, losses=losses,
                    top1=top1, top5=top5))
        writer.add_scalar('train/loss', loss.item(), cur_step)
        writer.add_scalar('train/top1', prec1.item(), cur_step)
        writer.add_scalar('train/top5', prec5.item(), cur_step)
        cur_step += 1

    logger.info("Train: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1.avg))
    return top1.avg 


def validate(valid_loader, model, epoch, cur_step):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    model.eval()

    with torch.no_grad():
        for step, (X, y) in enumerate(valid_loader):
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            N = X.size(0)

            logits = model(X)
            loss = model.criterion(logits, y)

            prec1, prec5 = utils.accuracy(logits, y, topk=(1, 5))
            losses.update(loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)

            if step % config.print_freq == 0 or step == len(valid_loader)-1:
                logger.info(
                    "Valid: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                        epoch+1, config.epochs, step, len(valid_loader)-1, losses=losses,
                        top1=top1, top5=top5))             
    writer.add_scalar('val/loss', losses.avg, cur_step)
    writer.add_scalar('val/top1', top1.avg, cur_step)
    writer.add_scalar('val/top5', top5.avg, cur_step)

    logger.info("Valid: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1.avg))

    return top1.avg

def validate_train(train_loader, model, epoch, cur_step):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    model.eval()

    with torch.no_grad():
        for step, (X, y) in enumerate(train_loader):
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            N = X.size(0)

            logits = model(X)
            loss = model.criterion(logits, y)

            prec1, prec5 = utils.accuracy(logits, y, topk=(1, 5))
            losses.update(loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)

            if step % config.print_freq == 0 or step == len(train_loader)-1:
                logger.info(
                    "Valid on training dataset: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                        epoch+1, config.epochs, step, len(train_loader)-1, losses=losses,
                        top1=top1, top5=top5))               
    logger.info("Valid on training dataset: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1.avg))

    return top1.avg

def adjust_learning_rate(optimizer, epoch):
    # """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
    # lr = config.w_lr * (0.5 ** (epoch // 30))
    """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
    lr = config.w_lr * (0.5 ** (epoch // config.w_decay_epoch))
    # """Sets the learning rate to the initial LR decayed by 2 every 15 epochs"""
    # lr = config.w_lr * (0.5 ** (epoch // 15))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



if __name__ == "__main__":
    main()
