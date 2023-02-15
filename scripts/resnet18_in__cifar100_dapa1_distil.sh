#### Train a ReLU pruned model with trainable mask, set the :

### increase lambda to speed up the replacement, otherwise it might not converge to high sparsity
### The lambda also controls the velocity of replacement, the middile velocity may lead to higher accuracy
mkdir -p ./out/


############### Best Accuracy Setting: ############### 
############ For higher ReLU count (higher or equal than 25), better to use lr = 0.001 and 200 epochs
############ For lower ReLU count (lower or equal than 15), better to decrease the learning rate, or number of epochs to prevent overfit
############ Low ReLU count comes with huge model redudancy, and can easily lead to overfit, so a good LR/epoch combination is important

# ########### Accuracy:  ###########
# nohup python -u train_cifar.py --gpu 0 --arch resnet18_in --ReLU_count 6 --w_mask_lr 0.0003\
#  --mask_epochs 150 --epochs 0 --degree 1 --lamda 20e1 --batch_size 128 --precision half --dataset cifar100\
#  --pretrained_path ./train_cifar/resnet18_in__cifar100/cosine_baseline_ReLUs0lr0.1ep400_baseline/best.pth.tar\
#  --distil --teacher_arch resnet18_in --teacher_path ./train_cifar/resnet18_in__cifar100/cosine_baseline_ReLUs0lr0.1ep400_baseline/best.pth.tar\
#  --optim cosine --act_type ReLU_masked_dapa_relay --threshold 0.003 > ./out/resnet18_in_mask_train_dapa1_cnt6_distil.txt &
# ### Location of checkpoint file, log file, and tb for tensorboard file are in folder: 
# ### 

# ########### Accuracy:  ###########
# nohup python -u train_cifar.py --gpu 2 --arch resnet18_in --ReLU_count 9 --w_mask_lr 0.0003\
#  --mask_epochs 150 --epochs 0 --degree 1 --lamda 14e1 --batch_size 128 --precision half --dataset cifar100\
#  --pretrained_path ./train_cifar/resnet18_in__cifar100/cosine_baseline_ReLUs0lr0.1ep400_baseline/best.pth.tar\
#  --distil --teacher_arch resnet18_in --teacher_path ./train_cifar/resnet18_in__cifar100/cosine_baseline_ReLUs0lr0.1ep400_baseline/best.pth.tar\
#  --optim cosine --act_type ReLU_masked_dapa_relay --threshold 0.003 > ./out/resnet18_in_mask_train_dapa1_cnt9_distil.txt &
# ### Location of checkpoint file, log file, and tb for tensorboard file are in folder: 
# ### 


# ########### Accuracy: 67.75 ###########
# nohup python -u train_cifar.py --gpu 0 --arch resnet18_in --ReLU_count 12.9 --w_mask_lr 0.0003\
#  --mask_epochs 150 --epochs 0 --degree 1 --lamda 8e1 --batch_size 128 --precision half --dataset cifar100\
#  --pretrained_path ./train_cifar/resnet18_in__cifar100/cosine_baseline_ReLUs0lr0.1ep400_baseline/best.pth.tar\
#  --distil --teacher_arch resnet18_in --teacher_path ./train_cifar/resnet18_in__cifar100/cosine_baseline_ReLUs0lr0.1ep400_baseline/best.pth.tar\
#  --optim cosine --act_type ReLU_masked_dapa_relay --threshold 0.003 > ./out/resnet18_in_mask_train_dapa1_cnt12.9_distil.txt &
# ### Location of checkpoint file, log file, and tb for tensorboard file are in folder: 
# ### 

# ########### Accuracy: 68.4 ###########
# nohup python -u train_cifar.py --gpu 1 --arch resnet18_in --ReLU_count 15 --w_mask_lr 0.001\
#  --mask_epochs 150 --epochs 0 --degree 1 --lamda 6e1 --batch_size 128 --precision half --dataset cifar100\
#  --pretrained_path ./train_cifar/resnet18_in__cifar100/cosine_baseline_ReLUs0lr0.1ep400_baseline/best.pth.tar\
#  --distil --teacher_arch resnet18_in --teacher_path ./train_cifar/resnet18_in__cifar100/cosine_baseline_ReLUs0lr0.1ep400_baseline/best.pth.tar\
#  --optim cosine --act_type ReLU_masked_dapa_relay --threshold 0.003 > ./out/resnet18_in_mask_train_dapa1_cnt15_distil.txt &
# ### Location of checkpoint file, log file, and tb for tensorboard file are in folder: 
# ### 

# ########### Accuracy: 71.10 ###########
# nohup python -u train_cifar.py --gpu 2 --arch resnet18_in --ReLU_count 25 --w_mask_lr 0.001\
#  --mask_epochs 200 --epochs 0 --degree 1 --lamda 4e1 --batch_size 128 --precision half --dataset cifar100 \
#  --pretrained_path ./train_cifar/resnet18_in__cifar100/cosine_baseline_ReLUs0lr0.1ep400_baseline/best.pth.tar\
#  --distil --teacher_arch resnet18_in --teacher_path ./train_cifar/resnet18_in__cifar100/cosine_baseline_ReLUs0lr0.1ep400_baseline/best.pth.tar\
#  --optim cosine --act_type ReLU_masked_dapa_relay --threshold 0.003 > ./out/resnet18_in_mask_train_dapa1_cnt25_distil.txt &
# ### Location of checkpoint file, log file, and tb for tensorboard file are in folder: 
# ### 


# ########### Accuracy: 73.86 ###########
# nohup python -u train_cifar.py --gpu 2 --arch resnet18_in --ReLU_count 50 --w_mask_lr 0.001\
#  --mask_epochs 200 --epochs 0 --degree 1 --lamda 4e1 --batch_size 128 --precision half --dataset cifar100 \
#  --pretrained_path ./train_cifar/resnet18_in__cifar100/cosine_baseline_ReLUs0lr0.1ep400_baseline/best.pth.tar\
#  --distil --teacher_arch resnet18_in --teacher_path ./train_cifar/resnet18_in__cifar100/cosine_baseline_ReLUs0lr0.1ep400_baseline/best.pth.tar\
#  --optim cosine --act_type ReLU_masked_dapa_relay --threshold 0.003 > ./out/resnet18_in_mask_train_dapa1_cnt50_distil.txt &
# ### Location of checkpoint file, log file, and tb for tensorboard file are in folder: 
# ### 

# ########### Accuracy:  ###########
# nohup python -u train_cifar.py --gpu 0 --arch resnet18_in --ReLU_count 80 --w_mask_lr 0.001\
#  --mask_epochs 200 --epochs 0 --degree 1 --lamda 4e1 --batch_size 128 --precision half --dataset cifar100 \
#  --pretrained_path ./train_cifar/resnet18_in__cifar100/cosine_baseline_ReLUs0lr0.1ep400_baseline/best.pth.tar\
#  --distil --teacher_arch resnet18_in --teacher_path ./train_cifar/resnet18_in__cifar100/cosine_baseline_ReLUs0lr0.1ep400_baseline/best.pth.tar\
#  --optim cosine --act_type ReLU_masked_dapa_relay --threshold 0.003 > ./out/resnet18_in_mask_train_dapa1_cnt80_distil.txt &
# ### Location of checkpoint file, log file, and tb for tensorboard file are in folder: 
# ### 

########### Accuracy:  ###########
nohup python -u train_cifar.py --gpu 2 --arch resnet18_in --ReLU_count 120 --w_mask_lr 0.001\
 --mask_epochs 200 --epochs 0 --degree 1 --lamda 4e1 --batch_size 128 --precision half --dataset cifar100 \
 --pretrained_path ./train_cifar/resnet18_in__cifar100/cosine_baseline_ReLUs0lr0.1ep400_baseline/best.pth.tar\
 --distil --teacher_arch resnet18_in --teacher_path ./train_cifar/resnet18_in__cifar100/cosine_baseline_ReLUs0lr0.1ep400_baseline/best.pth.tar\
 --optim cosine --act_type ReLU_masked_dapa_relay --threshold 0.003 > ./out/resnet18_in_mask_train_dapa1_cnt120_distil.txt &
### Location of checkpoint file, log file, and tb for tensorboard file are in folder: 
### 