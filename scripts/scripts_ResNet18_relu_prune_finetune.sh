#### Finetune a ResNet18 ReLU pruned model:
mkdir -p ./out/
nohup python -u train_cifar.py --gpu 0 --arch ResNet18 --sparsity 0.46 --mask_epochs 0 --mask_dropout 0.01\
 --w_lr 0.01 --w_weight_decay 1e-3 --epochs 100 \
 --pretrained_path ./train_cifar/ResNet18__cifar10/cosine_sp0.46wm_lr0.02mep100_baseline/checkpoint_mask_train.pth.tar\
 --optim cosine --act_type ReLU_masked_spgrad > ./out/ResNet18_mask_finetune_drop_sp0.46.txt &
### Location of checkpoint file, log file, and tb for tensorboard file are in folder: 
### train_cifar/ResNet18__cifar10/cosine_mask_dropout_0.01sp0.46lr0.01ep100_baseline

nohup python -u train_cifar.py --gpu 1 --arch ResNet18 --sparsity 0.64 --mask_epochs 0 --mask_dropout 0.01\
 --w_lr 0.01 --w_weight_decay 1e-3 --epochs 100 \
 --pretrained_path ./train_cifar/ResNet18__cifar10/cosine_sp0.64wm_lr0.02mep100_baseline/checkpoint_mask_train.pth.tar\
 --optim cosine --act_type ReLU_masked_spgrad > ./out/ResNet18_mask_finetune_drop_sp0.64.txt &
### Location of checkpoint file, log file, and tb for tensorboard file are in folder: 
### train_cifar/ResNet18__cifar10/cosine_mask_dropout_0.01sp0.64lr0.01ep100_baseline

nohup python -u train_cifar.py --gpu 2 --arch ResNet18 --sparsity 0.82 --mask_epochs 0 --mask_dropout 0.01\
 --w_lr 0.01 --w_weight_decay 1e-3 --epochs 100 \
 --pretrained_path ./train_cifar/ResNet18__cifar10/cosine_sp0.82wm_lr0.02mep100_baseline/checkpoint_mask_train.pth.tar\
 --optim cosine --act_type ReLU_masked_spgrad > ./out/ResNet18_mask_finetune_drop_sp0.82.txt &
### Location of checkpoint file, log file, and tb for tensorboard file are in folder: 
### train_cifar/ResNet18__cifar10/cosine_mask_dropout_0.01sp0.82lr0.01ep100_baseline

nohup python -u train_cifar.py --gpu 3 --arch ResNet18 --sparsity 0.91 --mask_epochs 0 --mask_dropout 0.01\
 --w_lr 0.01 --w_weight_decay 1e-3 --epochs 100 \
 --pretrained_path ./train_cifar/ResNet18__cifar10/cosine_sp0.91wm_lr0.02mep100_baseline/checkpoint_mask_train.pth.tar\
 --optim cosine --act_type ReLU_masked_spgrad > ./out/ResNet18_mask_finetune_drop_sp0.91.txt &
### Location of checkpoint file, log file, and tb for tensorboard file are in folder: 
### train_cifar/ResNet18__cifar10/cosine_mask_dropout_0.01sp0.91lr0.01ep100_baseline

nohup python -u train_cifar.py --gpu 4 --arch ResNet18 --sparsity 0.946 --mask_epochs 0 --mask_dropout 0.01\
 --w_lr 0.01 --w_weight_decay 1e-3 --epochs 100 \
 --pretrained_path ./train_cifar/ResNet18__cifar10/cosine_sp0.946wm_lr0.02mep100_baseline/checkpoint_mask_train.pth.tar\
 --optim cosine --act_type ReLU_masked_spgrad > ./out/ResNet18_mask_finetune_drop_sp0.946.txt &
### Location of checkpoint file, log file, and tb for tensorboard file are in folder: 
### train_cifar/ResNet18__cifar10/cosine_mask_dropout_0.01sp0.946lr0.01ep100_baseline

nohup python -u train_cifar.py --gpu 5 --arch ResNet18 --sparsity 0.964 --mask_epochs 0 --mask_dropout 0.01\
 --w_lr 0.01 --w_weight_decay 1e-3 --epochs 100 \
 --pretrained_path ./train_cifar/ResNet18__cifar10/cosine_sp0.964wm_lr0.02mep100_baseline/checkpoint_mask_train.pth.tar\
 --optim cosine --act_type ReLU_masked_spgrad > ./out/ResNet18_mask_finetune_drop_sp0.964.txt &
### Location of checkpoint file, log file, and tb for tensorboard file are in folder: 
### train_cifar/ResNet18__cifar10/cosine_mask_dropout_0.01sp0.964lr0.01ep100_baseline
