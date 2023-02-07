#### Train a ReLU pruned model with trainable mask, set the :

### increase lambda to speed up the replacement, otherwise it might not converge to high sparsity
### The lambda also controls the velocity of replacement, the middile velocity may lead to higher accuracy
mkdir -p ./out/
nohup python -u train_cifar.py --gpu 0 --arch resnet18 --ReLU_count 1.8 --w_mask_lr 0.01\
 --mask_epochs 50 --epochs 0 --degree 1 --lamda 4.0e1\
 --pretrained_path ./train_cifar/resnet18__cifar10/cosine_baseline_ReLUs0lr0.1ep400_baseline/best.pth.tar\
 --optim cosine --act_type ReLU_masked_autopoly_relay --threshold 0.003> ./out/resnet18_mask_train_autopoly_cnt1.8.txt &
### Location of checkpoint file, log file, and tb for tensorboard file are in folder: 
### train_cifar_autopoly1_relay/resnet18__cifar10_relay_0.003/cosine_ReLUs1.8wm_lr0.01mep50_baseline

nohup python -u train_cifar.py --gpu 1 --arch resnet18 --ReLU_count 12.9 --w_mask_lr 0.01\
 --mask_epochs 50 --epochs 0 --degree 1 --lamda 1.5e1\
 --pretrained_path ./train_cifar/resnet18__cifar10/cosine_baseline_ReLUs0lr0.1ep400_baseline/best.pth.tar\
 --optim cosine --act_type ReLU_masked_autopoly_relay --threshold 0.003> ./out/resnet18_mask_train_autopoly_cnt12.9.txt &
### Location of checkpoint file, log file, and tb for tensorboard file are in folder: 
### train_cifar_autopoly1_relay/resnet18__cifar10_relay_0.003/cosine_ReLUs12.9wm_lr0.01mep50_baseline


nohup python -u train_cifar.py --gpu 2 --arch resnet18 --ReLU_count 24.9 --w_mask_lr 0.01\
 --mask_epochs 50 --epochs 0 --degree 1 --lamda 1.3e1\
 --pretrained_path ./train_cifar/resnet18__cifar10/cosine_baseline_ReLUs0lr0.1ep400_baseline/best.pth.tar\
 --optim cosine --act_type ReLU_masked_autopoly_relay --threshold 0.003> ./out/resnet18_mask_train_autopoly_cnt24.9.txt &
### Location of checkpoint file, log file, and tb for tensorboard file are in folder: 
### train_cifar_autopoly1_relay/resnet18__cifar10_relay_0.003/cosine_ReLUs24.9wm_lr0.01mep50_baseline

nohup python -u train_cifar.py --gpu 3 --arch resnet18 --ReLU_count 51.2 --w_mask_lr 0.01\
 --mask_epochs 50 --epochs 0 --degree 1 --lamda 1.2e1\
 --pretrained_path ./train_cifar/resnet18__cifar10/cosine_baseline_ReLUs0lr0.1ep400_baseline/best.pth.tar\
 --optim cosine --act_type ReLU_masked_autopoly_relay --threshold 0.003> ./out/resnet18_mask_train_autopoly_cnt51.2.txt &
### Location of checkpoint file, log file, and tb for tensorboard file are in folder: 
### train_cifar_autopoly1_relay/resnet18__cifar10_relay_0.003/cosine_ReLUs51.2wm_lr0.01mep50_baseline

