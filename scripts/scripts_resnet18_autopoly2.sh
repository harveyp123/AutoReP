#### Train a ReLU pruned model with trainable mask, set the :
mkdir -p ./out/
nohup python -u train_cifar.py --gpu 0 --arch resnet18 --sparsity 0.5 --w_mask_lr 0.01 --mask_dropout 0.00 \
 --mask_epochs 50 --epochs 0 --degree 2\
 --pretrained_path ./train_cifar/resnet18__cifar10/cosine_baseline_sp0.0lr0.1ep400_baseline/best.pth.tar\
 --optim cosine --act_type ReLU_masked_autopoly_relay --threshold 0.003> ./out/resnet18_mask_train_autopoly_sp0.5.txt &
### Location of checkpoint file, log file, and tb for tensorboard file are in folder: 
### train_cifar_autopoly2_relay/resnet18__cifar10_relay_0.003/cosine_sp0.5wm_lr0.01mep50_baseline/cifar10.log

nohup python -u train_cifar.py --gpu 1 --arch resnet18 --sparsity 0.9 --w_mask_lr 0.01 --mask_dropout 0.00 \
 --mask_epochs 50 --epochs 0 --degree 2\
 --pretrained_path ./train_cifar/resnet18__cifar10/cosine_baseline_sp0.0lr0.1ep400_baseline/best.pth.tar\
 --optim cosine --act_type ReLU_masked_autopoly_relay --threshold 0.003> ./out/resnet18_mask_train_autopoly_sp0.9.txt &
### Location of checkpoint file, log file, and tb for tensorboard file are in folder: 
### train_cifar_autopoly2_relay/resnet18__cifar10_relay_0.003/cosine_sp0.9wm_lr0.01mep50_baseline/cifar10.log

nohup python -u train_cifar.py --gpu 2 --arch resnet18 --sparsity 0.95 --w_mask_lr 0.01 --mask_dropout 0.00 \
 --mask_epochs 50 --epochs 0 --degree 2\
 --pretrained_path ./train_cifar/resnet18__cifar10/cosine_baseline_sp0.0lr0.1ep400_baseline/best.pth.tar\
 --optim cosine --act_type ReLU_masked_autopoly_relay --threshold 0.003> ./out/resnet18_mask_train_autopoly_sp0.95.txt &
### Location of checkpoint file, log file, and tb for tensorboard file are in folder: 
### train_cifar_autopoly2_relay/resnet18__cifar10_relay_0.003/cosine_sp0.95wm_lr0.01mep50_baseline/cifar10.log
