#### Train a ReLU pruned model with trainable mask, set the :

### increase lambda to speed up the replacement, otherwise it might not converge to high sparsity
### The lambda also controls the velocity of replacement, the middile velocity may lead to higher accuracy
mkdir -p ./out/
nohup python -u train_cifar.py --gpu 1 --arch resnet18 --ReLU_count 1.8 --w_mask_lr 0.01 --w_lr 0.002\
 --mask_epochs 0 --epochs 50 --degree 2 --lamda 6.0e1 --scale_x2 0.2\
 --pretrained_path ./train_cifar_dapa2_relay/resnet18__cifar10_relay_0.003/cosine_ReLUs1.8wm_lr0.01mep50_baseline/checkpoint_mask_train.pth.tar\
 --optim cosine --act_type ReLU_masked_dapa_relay --threshold 0.003> ./out/resnet18_mask_train_dapa2_cnt1.8_finetune.txt &
### Location of checkpoint file, log file, and tb for tensorboard file are in folder: 
### 

nohup python -u train_cifar.py --gpu 3 --arch resnet18 --ReLU_count 12.9 --w_mask_lr 0.01 --w_lr 0.002\
 --mask_epochs 0 --epochs 50 --degree 2 --lamda 2.0e1 --scale_x2 0.2\
 --pretrained_path ./train_cifar_dapa2_relay/resnet18__cifar10_relay_0.003/cosine_ReLUs12.9wm_lr0.01mep50_baseline/checkpoint_mask_train.pth.tar\
 --optim cosine --act_type ReLU_masked_dapa_relay --threshold 0.003> ./out/resnet18_mask_train_dapa2_cnt12.9_finetune.txt &
### Location of checkpoint file, log file, and tb for tensorboard file are in folder: 
### 

nohup python -u train_cifar.py --gpu 5 --arch resnet18 --ReLU_count 24.9 --w_mask_lr 0.01 --w_lr 0.002\
 --mask_epochs 0 --epochs 50 --degree 2 --lamda 1.5e1 --scale_x2 0.2\
 --pretrained_path ./train_cifar_dapa2_relay/resnet18__cifar10_relay_0.003/cosine_ReLUs24.9wm_lr0.01mep50_baseline/checkpoint_mask_train.pth.tar\
 --optim cosine --act_type ReLU_masked_dapa_relay --threshold 0.003> ./out/resnet18_mask_train_dapa2_cnt24.9_finetune.txt &
### Location of checkpoint file, log file, and tb for tensorboard file are in folder: 
### 

nohup python -u train_cifar.py --gpu 6 --arch resnet18 --ReLU_count 51.2 --w_mask_lr 0.01 --w_lr 0.002\
 --mask_epochs 0 --epochs 50 --degree 2 --lamda 1.5e1 --scale_x2 0.2\
 --pretrained_path ./train_cifar_dapa2_relay/resnet18__cifar10_relay_0.003/cosine_ReLUs51.2wm_lr0.01mep50_baseline/checkpoint_mask_train.pth.tar\
 --optim cosine --act_type ReLU_masked_dapa_relay --threshold 0.003> ./out/resnet18_mask_train_dapa2_cnt51.2_finetune.txt &
### Location of checkpoint file, log file, and tb for tensorboard file are in folder: 
### 

