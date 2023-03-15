import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import util_func.utils as utils
from models_util import *
import time

def validate(valid_loader, model, epoch, cur_step, device, config, logger, writer):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    model.eval()
    #model.train()

    with torch.no_grad():
        for step, (X, y) in enumerate(valid_loader):
            N = X.size(0)
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            # print("X[0]: ", X[0])
            # print("X[1]: ", X[1])
            # X, y = X.to(device), y.to(device)
            # X = X.cuda()
            # y = y.cuda()
            # print("X[0]: ", X[0])
            # print("X[1]: ", X[1])
            # torch.cuda.synchronize()
            logits = model(X)
            # time. sleep(1)
            # torch.cuda.synchronize()
            loss = model.criterion(logits, y)
            # torch.cuda.synchronize()
            # print("logits: ", logits)
            # print("y: ", y)
            # time. sleep(1)
            # print("logits: ", logits)
            # print("y: ", y)
            prec1, prec5 = utils.accuracy(logits.cpu(), y.cpu(), topk=(1, 5))
            # torch.cuda.synchronize()
            #print("top1 accuracy: ", prec1)
            #print("top1 accuracy: ", prec5)
            # exit()
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

def validate_train(train_loader, model, epoch, cur_step, device, config, logger, writer):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    model.eval()

    with torch.no_grad():
        for step, (X, y) in enumerate(train_loader):
            N = X.size(0)
            # print("X:", X)
            # print("y:", y)
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            # X, y = X.to(device), y.to(device)
            # X = X.cuda()
            # y = y.cuda()
            # print("X:", X)
            # print("y:", y)
            # torch.cuda.synchronize()
            logits = model(X)
            # torch.cuda.synchronize()
            loss = model.criterion(logits, y)
            # torch.cuda.synchronize()
            prec1, prec5 = utils.accuracy(logits.cpu(), y.cpu(), topk=(1, 5))
            # torch.cuda.synchronize()
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

def validate_fp16(valid_loader, model, epoch, cur_step, device, config, logger, writer):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()
    use_amp = True
    model.eval()
    #model.train()

    with torch.no_grad():
        for step, (X, y) in enumerate(valid_loader):
            N = X.size(0)
            # X = X.cuda()
            # y = y.cuda()
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            # torch.cuda.synchronize()
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                logits = model(X)
                # torch.cuda.synchronize()
                loss = model.criterion(logits, y)
                # torch.cuda.synchronize()
            logits = logits.float()
            loss = loss.float()
            prec1, prec5 = utils.accuracy(logits.cpu(), y.cpu(), topk=(1, 5))
            # torch.cuda.synchronize()
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

def validate_train_fp16(train_loader, model, epoch, cur_step, device, config, logger, writer):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()
    use_amp = True
    model.eval()

    with torch.no_grad():
        for step, (X, y) in enumerate(train_loader):
            N = X.size(0)
            # X = X.cuda()
            # y = y.cuda()
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            # torch.cuda.synchronize()
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                logits = model(X)
                # torch.cuda.synchronize()
                loss = model.criterion(logits, y)
                # torch.cuda.synchronize()
            logits = logits.float()
            loss = loss.float()
            prec1, prec5 = utils.accuracy(logits.cpu(), y.cpu(), topk=(1, 5))
            # torch.cuda.synchronize()
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





# # ----------------------------------------------
# from enum import Enum
# class Summary(Enum):
#     NONE = 0
#     AVERAGE = 1
#     SUM = 2
#     COUNT = 3

# class AverageMeter(object):
#     """Computes and stores the average and current value"""
#     def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
#         self.name = name
#         self.fmt = fmt
#         self.summary_type = summary_type
#         self.reset()

#     def reset(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0

#     def update(self, val, n=1):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / self.count

#     def __str__(self):
#         fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
#         return fmtstr.format(**self.__dict__)
    
#     def summary(self):
#         fmtstr = ''
#         if self.summary_type is Summary.NONE:
#             fmtstr = ''
#         elif self.summary_type is Summary.AVERAGE:
#             fmtstr = '{name} {avg:.3f}'
#         elif self.summary_type is Summary.SUM:
#             fmtstr = '{name} {sum:.3f}'
#         elif self.summary_type is Summary.COUNT:
#             fmtstr = '{name} {count:.3f}'
#         else:
#             raise ValueError('invalid summary type %r' % self.summary_type)
        
#         return fmtstr.format(**self.__dict__)


# class ProgressMeter(object):
#     def __init__(self, num_batches, meters, prefix=""):
#         self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
#         self.meters = meters
#         self.prefix = prefix

#     def display(self, batch):
#         entries = [self.prefix + self.batch_fmtstr.format(batch)]
#         entries += [str(meter) for meter in self.meters]
#         print('\t'.join(entries))
        
#     def display_summary(self):
#         entries = [" *"]
#         entries += [meter.summary() for meter in self.meters]
#         print(' '.join(entries))

#     def _get_batch_fmtstr(self, num_batches):
#         num_digits = len(str(num_batches // 1))
#         fmt = '{:' + str(num_digits) + 'd}'
#         return '[' + fmt + '/' + fmt.format(num_batches) + ']'

# def accuracy(output, target, topk=(1,)):
#     """Computes the accuracy over the k top predictions for the specified values of k"""
#     with torch.no_grad():
#         maxk = max(topk)
#         batch_size = target.size(0)

#         _, pred = output.topk(maxk, 1, True, True)
#         pred = pred.t()
#         correct = pred.eq(target.view(1, -1).expand_as(pred))

#         res = []
#         for k in topk:
#             correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
#             res.append(correct_k.mul_(100.0 / batch_size))
#         return res
# def test_val(val_loader, model, epoch, cur_step, device, config, logger, writer):
#     batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
#     losses = AverageMeter('Loss', ':.4e', Summary.NONE)
#     top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
#     top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
#     progress = ProgressMeter(
#         len(val_loader),
#         [batch_time, losses, top1, top5],
#         prefix='Test: ')

#     # switch to evaluate mode
#     model.eval()
#     with torch.no_grad():
#         end = time.time()
#         for i, (images, target) in enumerate(val_loader):
#             images = images.cuda()
#             target = target.cuda()

#             # compute output
#             output = model(images)
#             loss = model.criterion(output, target)

#             # measure accuracy and record loss
#             acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
#             losses.update(loss.item(), images.size(0))
#             top1.update(acc1.item(), images.size(0))
#             top5.update(acc5.item(), images.size(0))

#             # measure elapsed time
#             batch_time.update(time.time() - end)
#             end = time.time()

#             if i % config.print_freq == 0:
#                 logger.info(
#                     "Valid: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
#                     "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
#                         epoch+1, config.epochs, i, len(val_loader)-1, losses=losses,
#                         top1=top1, top5=top5))
#         progress.display_summary()
    
#     logger.info("Valid: Final Prec@1 {:.4%}, Prec@5 {:.4%}".format(top1.avg, top5.avg))


#     return top1.avg, top5.avg


