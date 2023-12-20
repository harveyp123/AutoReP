mkdir -p ./out/


nohup python -u train_cifar.py --gpu 1 --arch resnet18_in --ReLU_count 9 --w_mask_lr 0.001\
 --degree 2 --scale_x2 0.1 \
 --mask_epochs 200 --epochs 0  --lamda 14e1 --batch_size 128 --precision full --dataset cifar100\
 --pretrained_path ./train_cifar/resnet18_in__cifar100/cosine_baseline_ReLUs0lr0.1ep400_baseline/best.pth.tar\
 --distil --teacher_arch resnet18_in --teacher_path ./train_cifar/resnet18_in__cifar100/cosine_baseline_ReLUs0lr0.1ep400_baseline/best.pth.tar\
 --optim cosine --act_type ReLU_masked_dapa_relay --threshold 0.003 > ./out/resnet18_in_mask_train_dapa2_cnt9_distil.txt &
### Location of checkpoint file, log file, and tb for tensorboard file are in folder: 
### 

nohup python -u train_cifar.py --gpu 0 --arch wide_resnet_22_8 --ReLU_count 80 --w_mask_lr 0.001\
 --degree 2 --scale_x2 0.1 \
 --mask_epochs 200 --epochs 0 --lamda 8e1 --batch_size 128 --precision full --dataset cifar100\
 --pretrained_path ./train_cifar/wide_resnet_22_8__cifar100/cosine_baseline_ReLUs0lr0.1ep400_baseline/best.pth.tar\
 --distil --teacher_arch wide_resnet_22_8 --teacher_path ./train_cifar/wide_resnet_22_8__cifar100/cosine_baseline_ReLUs0lr0.1ep400_baseline/best.pth.tar\
 --optim cosine --act_type ReLU_masked_dapa_relay --threshold 0.003 > ./out/wide_resnet_22_8_mask_train_dapa1_cnt80_distil.txt &
