#### Train a ReLU pruned model with trainable mask:
mkdir -p ./out/
nohup python -u train_cifar.py --gpu 2 --arch resnet18 --sparsity 0.5 --w_mask_lr 0.02 --mask_dropout 0.01 \
 --mask_epochs 100 --epochs 0 --threshold 0.001\
 --pretrained_path ./train_cifar/resnet18__cifar10/cosine_baseline_sp0.0lr0.1ep400_baseline/best.pth.tar\
 --optim cosine --act_type ReLU_masked_spgrad_relay > ./out/resnet18_mask_train_drop_relay_sp0.5.txt &
### Location of checkpoint file, log file, and tb for tensorboard file are in folder: 
### train_cifar/resnet18__cifar10/cosine_mask_dropout_0.01sp0.5wm_lr0.02mep100_baseline

nohup python -u train_cifar.py --gpu 3 --arch resnet18 --sparsity 0.9 --w_mask_lr 0.02 --mask_dropout 0.01 \
 --mask_epochs 100 --epochs 0 --threshold 0.001\
 --pretrained_path ./train_cifar/resnet18__cifar10/cosine_baseline_sp0.0lr0.1ep400_baseline/best.pth.tar\
 --optim cosine --act_type ReLU_masked_spgrad_relay > ./out/resnet18_mask_train_drop_relay_sp0.9.txt &
### Location of checkpoint file, log file, and tb for tensorboard file are in folder: 
### train_cifar/resnet18__cifar10/cosine_mask_dropout_0.01sp0.9wm_lr0.02mep100_baseline

nohup python -u train_cifar.py --gpu 4 --arch resnet18 --sparsity 0.5 --w_mask_lr 0.02 --mask_dropout 0.00 \
 --mask_epochs 100 --epochs 0 --threshold 0.001\
 --pretrained_path ./train_cifar/resnet18__cifar10/cosine_baseline_sp0.0lr0.1ep400_baseline/best.pth.tar\
 --optim cosine --act_type ReLU_masked_spgrad_relay > ./out/resnet18_mask_train_relay_sp0.5.txt &
### Location of checkpoint file, log file, and tb for tensorboard file are in folder: 
### train_cifar/resnet18__cifar10/cosine_sp0.5wm_lr0.02mep100_baseline

nohup python -u train_cifar.py --gpu 5 --arch resnet18 --sparsity 0.9 --w_mask_lr 0.02 --mask_dropout 0.00 \
 --mask_epochs 100 --epochs 0 --threshold 0.001\
 --pretrained_path ./train_cifar/resnet18__cifar10/cosine_baseline_sp0.0lr0.1ep400_baseline/best.pth.tar\
 --optim cosine --act_type ReLU_masked_spgrad_relay > ./out/resnet18_mask_train_relay_sp0.9.txt &
### Location of checkpoint file, log file, and tb for tensorboard file are in folder: 
### train_cifar/resnet18__cifar10/cosine_sp0.9wm_lr0.02mep100_baseline