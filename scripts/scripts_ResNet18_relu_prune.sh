#### Train a ReLU pruned model (ResNet18) with trainable mask:
mkdir -p ./out/
nohup python -u train_cifar.py --gpu 0 --arch ResNet18 --sparsity 0.46 --w_mask_lr 0.02 \
 --mask_epochs 100 --epochs 0 \
 --pretrained_path ./train_cifar/ResNet18__cifar10/cosine_baseline_sp0.0lr0.1ep400_baseline/best.pth.tar\
 --optim cosine --act_type ReLU_masked_spgrad > ./out/ResNet18_mask_train_drop_sp0.46.txt &
### Location of checkpoint file, log file, and tb for tensorboard file are in folder: 
### train_cifar/ResNet18__cifar10/cosine_sp0.46wm_lr0.02mep100_baseline

nohup python -u train_cifar.py --gpu 1 --arch ResNet18 --sparsity 0.64 --w_mask_lr 0.02 \
 --mask_epochs 100 --epochs 0 \
 --pretrained_path ./train_cifar/ResNet18__cifar10/cosine_baseline_sp0.0lr0.1ep400_baseline/best.pth.tar\
 --optim cosine --act_type ReLU_masked_spgrad > ./out/ResNet18_mask_train_drop_sp0.64.txt &
### Location of checkpoint file, log file, and tb for tensorboard file are in folder: 
### train_cifar/ResNet18__cifar10/cosine_sp0.64wm_lr0.02mep100_baseline

nohup python -u train_cifar.py --gpu 2 --arch ResNet18 --sparsity 0.82 --w_mask_lr 0.02 \
 --mask_epochs 100 --epochs 0 \
 --pretrained_path ./train_cifar/ResNet18__cifar10/cosine_baseline_sp0.0lr0.1ep400_baseline/best.pth.tar\
 --optim cosine --act_type ReLU_masked_spgrad > ./out/ResNet18_mask_train_drop_sp0.82.txt &
### Location of checkpoint file, log file, and tb for tensorboard file are in folder: 
### train_cifar/ResNet18__cifar10/cosine_sp0.82wm_lr0.02mep100_baseline

nohup python -u train_cifar.py --gpu 3 --arch ResNet18 --sparsity 0.91 --w_mask_lr 0.02 \
 --mask_epochs 100 --epochs 0 \
 --pretrained_path ./train_cifar/ResNet18__cifar10/cosine_baseline_sp0.0lr0.1ep400_baseline/best.pth.tar\
 --optim cosine --act_type ReLU_masked_spgrad > ./out/ResNet18_mask_train_drop_sp0.91.txt &
### Location of checkpoint file, log file, and tb for tensorboard file are in folder: 
### train_cifar/ResNet18__cifar10/cosine_sp0.91wm_lr0.02mep100_baseline

nohup python -u train_cifar.py --gpu 4 --arch ResNet18 --sparsity 0.946 --w_mask_lr 0.02 \
 --mask_epochs 100 --epochs 0 \
 --pretrained_path ./train_cifar/ResNet18__cifar10/cosine_baseline_sp0.0lr0.1ep400_baseline/best.pth.tar\
 --optim cosine --act_type ReLU_masked_spgrad > ./out/ResNet18_mask_train_drop_sp0.946.txt &
### Location of checkpoint file, log file, and tb for tensorboard file are in folder: 
### train_cifar/ResNet18__cifar10/cosine_sp0.946wm_lr0.02mep100_baseline

nohup python -u train_cifar.py --gpu 5 --arch ResNet18 --sparsity 0.964 --w_mask_lr 0.02 \
 --mask_epochs 100 --epochs 0 \
 --pretrained_path ./train_cifar/ResNet18__cifar10/cosine_baseline_sp0.0lr0.1ep400_baseline/best.pth.tar\
 --optim cosine --act_type ReLU_masked_spgrad > ./out/ResNet18_mask_train_drop_sp0.964.txt &
### Location of checkpoint file, log file, and tb for tensorboard file are in folder: 
### train_cifar/ResNet18__cifar10/cosine_sp0.964wm_lr0.02mep100_baseline