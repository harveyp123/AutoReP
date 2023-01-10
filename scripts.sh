# nohup python -u train_cifar.py --gpu 4 --act_type nn.ReLU > resnet18_baseline.txt &
# nohup python -u train_cifar.py --gpu 3 --epochs 255 --optim cosine_rst --act_type nn.ReLU > resnet18_baseline_cosine_rst.txt &
# nohup python -u train_cifar.py --gpu 3 --w_lr 0.08 --epochs 255 --optim cosine_rst --act_type nn.ReLU > resnet18_baseline_cosine_rst_lr0.08.txt &
# nohup python -u train_cifar.py --gpu 4 --w_lr 0.12 --epochs 255 --optim cosine_rst --act_type nn.ReLU > resnet18_baseline_cosine_rst_lr0.12.txt &
# nohup python -u train_cifar.py --gpu 1 --w_lr 0.0601 --epochs 255 --optim cosine_rst --act_type nn.ReLU > resnet18_baseline_cosine_rst_lr0.06.txt &
# nohup python -u train_cifar.py --mask_epochs 35 --pretrained_path /data3/hop20001/MPC_sparse_act/ACT_prune_cleaned/train_cifar/ResNet18_cifar10/baseline_lr0.1ep200_baseline/best.pth.tar\
#         --gpu 4 --act_type ReLU_masked --lamda 1e0 > resnet18_lamda_1.txt &
# nohup python -u train_cifar.py --mask_epochs 40 --pretrained_path /data3/hop20001/MPC_sparse_act/ACT_prune_cleaned/train_cifar/ResNet18_cifar10/baseline_lr0.1ep200_baseline/best.pth.tar\
#         --gpu 3 --act_type ReLU_masked --optim cosine_rst --lamda 1e0 > resnet18_lamda_1_cos_rst.txt &

# nohup python -u train_cifar.py --mask_epochs 40  --epochs 240 --pretrained_path /data3/hop20001/MPC_sparse_act/ACT_prune_cleaned/train_cifar/ResNet18_cifar10/baseline_lr0.1ep200_baseline/best.pth.tar\
#         --gpu 0 --act_type ReLU_masked --optim cosine_rst --lamda 1e0 > resnet18_lamda_1_cos_rst_240epochs.txt &

# nohup python -u train_cifar.py --mask_epochs 40  --epochs 240 \
#         --arch resnet18 --gpu 0 --act_type ReLU_masked --optim cosine_rst --lamda 1e0 > resnet18_base_lamda_1_cos_rst_240epochs.txt &

# nohup python -u train_cifar.py --epochs 200 --seed 9 --w_lr 0.101 \
#         --arch resnet18 --gpu 1 --act_type nn.ReLU --optim cosine > resnet18_base_cos_200epochs.txt &

# nohup python -u train_cifar.py --epochs 200 --seed 9 --w_lr 0.0801 \
#         --arch resnet18 --gpu 1 --act_type nn.ReLU --optim cosine > resnet18_base_cos_200epochs_lr0.08.txt &

# nohup python -u train_cifar.py --epochs 200 --seed 9 --w_lr 0.15 \
#         --arch resnet18 --gpu 6 --act_type nn.ReLU --optim cosine > resnet18_base_cos_200epochs_lr0.15.txt &

nohup python -u train_cifar.py --mask_epochs 40  --epochs 240 \
        --arch resnet18 --gpu 0 --w_lr 0.08 --act_type ReLU_masked --optim cosine_rst --lamda 1e0 > resnet18_base_lamda_1_cos_rst_240epochs_lr0.08.txt &

nohup python -u train_cifar.py --mask_epochs 40  --epochs 240 \
        --arch resnet18 --gpu 1 --w_lr 0.12 --act_type ReLU_masked --optim cosine_rst --lamda 1e0 > resnet18_base_lamda_1_cos_rst_240epochs_lr0.12.txt &