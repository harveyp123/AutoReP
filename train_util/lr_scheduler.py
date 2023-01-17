import math
def cos_modified_learning_rate(optimizer, epoch, config):
    T_start = config.epochs
    T_converge = config.epochs*4//5
    lr_min = config.w_lr_min
    lr_base = config.w_lr
    if epoch < T_start:
        lr = lr_min + (lr_base - lr_min) * \
               (1 + math.cos((epoch) * math.pi / (T_start * 2))) / 2
    else:
        lr = lr_min + (lr_base - lr_min) * \
               (1 + math.cos((epoch + T_converge - T_start) * math.pi / (T_converge * 2))) / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr