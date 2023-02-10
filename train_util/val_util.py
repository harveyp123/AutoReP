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
            # X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            # print("X[0]: ", X[0])
            # print("X[1]: ", X[1])
            # X, y = X.to(device), y.to(device)
            X = X.cuda()
            y = y.cuda()
            # print("X[0]: ", X[0])
            # print("X[1]: ", X[1])
            torch.cuda.synchronize()
            logits = model(X)
            # time. sleep(1)
            torch.cuda.synchronize()
            loss = model.criterion(logits, y)
            torch.cuda.synchronize()
            # print("logits: ", logits)
            # print("y: ", y)
            # time. sleep(1)
            # print("logits: ", logits)
            # print("y: ", y)
            prec1, prec5 = utils.accuracy(logits.cpu(), y.cpu(), topk=(1, 5))
            torch.cuda.synchronize()
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
            # X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            # X, y = X.to(device), y.to(device)
            X = X.cuda()
            y = y.cuda()
            # print("X:", X)
            # print("y:", y)
            torch.cuda.synchronize()
            logits = model(X)
            torch.cuda.synchronize()
            loss = model.criterion(logits, y)
            torch.cuda.synchronize()
            prec1, prec5 = utils.accuracy(logits.cpu(), y.cpu(), topk=(1, 5))
            torch.cuda.synchronize()
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