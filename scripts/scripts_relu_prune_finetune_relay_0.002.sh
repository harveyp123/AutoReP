#### Finetune a ReLU pruned model:
mkdir -p ./out/
nohup python -u train_cifar.py --gpu 0 --arch resnet18 --sparsity 0.5 --mask_epochs 0 --mask_dropout 0.01\
 --w_lr 0.01 --w_weight_decay 1e-3 --epochs 100  --threshold 0.002\
 --pretrained_path ./train_cifar_relay/resnet18__cifar10_relay_0.002/cosine_mask_dropout_0.01sp0.5wm_lr0.02mep100_baseline/checkpoint_mask_train.pth.tar\
 --optim cosine --act_type ReLU_masked_spgrad_relay > ./out/resnet18_mask_finetune_drop_sp0.5.txt &
### Location of checkpoint file, log file, and tb for tensorboard file are in folder: 
### train_cifar/resnet18__cifar10/cosine_mask_dropout_0.01sp0.5lr0.01ep100_baseline

nohup python -u train_cifar.py --gpu 1 --arch resnet18 --sparsity 0.9 --mask_epochs 0 --mask_dropout 0.01\
 --w_lr 0.01 --w_weight_decay 1e-3 --epochs 100  --threshold 0.002\
 --pretrained_path ./train_cifar_relay/resnet18__cifar10_relay_0.002/cosine_mask_dropout_0.01sp0.9wm_lr0.02mep100_baseline/checkpoint_mask_train.pth.tar\
 --optim cosine --act_type ReLU_masked_spgrad_relay > ./out/resnet18_mask_finetune_drop_sp0.9.txt &
### Location of checkpoint file, log file, and tb for tensorboard file are in folder: 
### train_cifar/resnet18__cifar10/cosine_mask_dropout_0.01sp0.9lr0.01ep100_baseline

