#### Train a ReLU pruned model with trainable mask, set the :
mkdir -p ./out/
nohup python -u train_cifar.py --gpu 0 --arch ResNet18 --sparsity 0.91 --w_mask_lr 0.01 --mask_dropout 0.00 \
 --mask_epochs 50 --epochs 0 --scale_x2 0.1 --scale_x 0.5\
 --pretrained_path ./train_cifar/ResNet18__cifar10/cosine_baseline_sp0.0lr0.1ep400_baseline/best.pth.tar\
 --optim cosine --act_type ReLU_masked_poly > ./out/ResNet18_mask_train_poly_sp0.91.txt &
### Location of checkpoint file, log file, and tb for tensorboard file are in folder: 
### train_cifar_poly/ResNet18__cifar10/cosine_sp0.91wm_lr0.01mep50_baseline

nohup python -u train_cifar.py --gpu 1 --arch ResNet18 --sparsity 0.946 --w_mask_lr 0.01 --mask_dropout 0.00 \
 --mask_epochs 50 --epochs 0 --scale_x2 0.1 --scale_x 0.5\
 --pretrained_path ./train_cifar/ResNet18__cifar10/cosine_baseline_sp0.0lr0.1ep400_baseline/best.pth.tar\
 --optim cosine --act_type ReLU_masked_poly > ./out/ResNet18_mask_train_poly_sp0.946.txt &
### Location of checkpoint file, log file, and tb for tensorboard file are in folder: 
### train_cifar_poly/ResNet18__cifar10/cosine_sp0.946wm_lr0.01mep50_baseline

nohup python -u train_cifar.py --gpu 2 --arch ResNet18 --sparsity 0.964 --w_mask_lr 0.01 --mask_dropout 0.00 \
 --mask_epochs 50 --epochs 0 --scale_x2 0.1 --scale_x 0.5\
 --pretrained_path ./train_cifar/ResNet18__cifar10/cosine_baseline_sp0.0lr0.1ep400_baseline/best.pth.tar\
 --optim cosine --act_type ReLU_masked_poly > ./out/ResNet18_mask_train_poly_sp0.964.txt &
### Location of checkpoint file, log file, and tb for tensorboard file are in folder: 
### train_cifar_poly/ResNet18__cifar10/cosine_sp0.964wm_lr0.01mep50_baseline

# nohup python -u train_cifar.py --gpu 0 --arch ResNet18 --sparsity 0.91 --w_mask_lr 0.02 --mask_dropout 0.00 \
#  --mask_epochs 100 --epochs 0 --scale_x2 0.1 --scale_x 0.5\
#  --pretrained_path ./train_cifar/ResNet18__cifar10/cosine_baseline_sp0.0lr0.1ep400_baseline/best.pth.tar\
#  --optim cosine --act_type ReLU_masked_poly > ./out/ResNet18_mask_train_poly_sp0.91.txt &
# ### Location of checkpoint file, log file, and tb for tensorboard file are in folder: 
# ### train_cifar_poly/ResNet18__cifar10/cosine_sp0.91wm_lr0.02mep100_baseline

# nohup python -u train_cifar.py --gpu 1 --arch ResNet18 --sparsity 0.946 --w_mask_lr 0.02 --mask_dropout 0.00 \
#  --mask_epochs 100 --epochs 0 --scale_x2 0.1 --scale_x 0.5\
#  --pretrained_path ./train_cifar/ResNet18__cifar10/cosine_baseline_sp0.0lr0.1ep400_baseline/best.pth.tar\
#  --optim cosine --act_type ReLU_masked_poly > ./out/ResNet18_mask_train_poly_sp0.946.txt &
# ### Location of checkpoint file, log file, and tb for tensorboard file are in folder: 
# ### train_cifar_poly/ResNet18__cifar10/cosine_sp0.946wm_lr0.02mep100_baseline

# nohup python -u train_cifar.py --gpu 2 --arch ResNet18 --sparsity 0.964 --w_mask_lr 0.02 --mask_dropout 0.00 \
#  --mask_epochs 100 --epochs 0 --scale_x2 0.1 --scale_x 0.5\
#  --pretrained_path ./train_cifar/ResNet18__cifar10/cosine_baseline_sp0.0lr0.1ep400_baseline/best.pth.tar\
#  --optim cosine --act_type ReLU_masked_poly > ./out/ResNet18_mask_train_poly_sp0.964.txt &
# ### Location of checkpoint file, log file, and tb for tensorboard file are in folder: 
# ### train_cifar_poly/ResNet18__cifar10/cosine_sp0.964wm_lr0.02mep100_baseline