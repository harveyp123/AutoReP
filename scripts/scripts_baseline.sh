#### Train a baseline model with ReLU activation function:
mkdir -p ./out/
# nohup python -u train_cifar.py --gpu 0 --arch resnet9 --w_lr 0.1 --mask_epochs 0 --epochs 400 \
#  --optim cosine --act_type nn.ReLU > ./out/resnet9_baseline.txt &
# ### Location of checkpoint file, log file, and tb for tensorboard file are in folder: 
# ### train_cifar/resnet9__cifar10/cosine_baseline_ReLUs0lr0.1ep400_baseline

# nohup python -u train_cifar.py --gpu 1 --arch resnet18 --w_lr 0.1 --mask_epochs 0 --epochs 400 \
#  --optim cosine --act_type nn.ReLU > ./out/resnet18_baseline.txt &
# ### Location of checkpoint file, log file, and tb for tensorboard file are in folder: 
# ### train_cifar/resnet18__cifar10/cosine_baseline_ReLUs0lr0.1ep400_baseline

# nohup python -u train_cifar.py --gpu 2 --arch ResNet18 --w_lr 0.1 --mask_epochs 0 --epochs 400 \
#  --optim cosine --act_type nn.ReLU > ./out/ResNet18_baseline.txt &
# ### Location of checkpoint file, log file, and tb for tensorboard file are in folder: 
# ### train_cifar/ResNet18__cifar10/cosine_baseline_ReLUs0lr0.1ep400_baseline

# nohup python -u train_cifar.py --gpu 5 --arch resnet18 --w_lr 0.1 --mask_epochs 0 --epochs 400 --dataset cifar100\
#  --optim cosine --act_type nn.ReLU > ./out/resnet18_cifar100_baseline.txt &
# ### Location of checkpoint file, log file, and tb for tensorboard file are in folder: 
# ### train_cifar/resnet9__cifar10/cosine_baseline_ReLUs0lr0.1ep400_baseline

# nohup python -u train_cifar.py --gpu 3 --arch resnet18_in --w_lr 0.1 --mask_epochs 0 --epochs 400 --dataset cifar10\
#  --optim cosine --act_type nn.ReLU > ./out/resnet18_in_cifar10_baseline.txt &
# nohup python -u train_cifar.py --gpu 0 --arch resnet18_in --w_lr 0.1 --mask_epochs 0 --epochs 400 --dataset cifar100\
#  --optim cosine --act_type nn.ReLU > ./out/resnet18_in_cifar100_baseline.txt &
# nohup python -u train_cifar.py --gpu 5 --arch resnet18_in --w_lr 0.1 --mask_epochs 0 --epochs 400 --dataset tiny_imagenet --data_path "/data/tiny-imagenet-200"\
#  --optim cosine --act_type nn.ReLU > ./out/resnet18_in_tiny_imagenet_baseline.txt &

# nohup python -u train_cifar.py --gpu 2 --arch wide_resnet_22_8 --w_lr 0.1 --mask_epochs 0 --epochs 400 --dataset cifar10\
#  --optim cosine --act_type nn.ReLU > ./out/wide_resnet_22_8_cifar10_baseline.txt &
# nohup python -u train_cifar.py --gpu 1 --arch wide_resnet_22_8 --w_lr 0.1 --mask_epochs 0 --epochs 400 --dataset cifar100\
#  --optim cosine --act_type nn.ReLU > ./out/wide_resnet_22_8_cifar100_baseline.txt &
nohup python -u train_cifar.py --gpu 0 --arch wide_resnet_22_8 --w_lr 0.1 --mask_epochs 0 --epochs 400 --dataset tiny_imagenet --data_path "/data/tiny-imagenet-200"\
 --optim cosine --act_type nn.ReLU > ./out/wide_resnet_22_8_tiny_imagenet_baseline.txt &

# nohup python -u train_cifar.py --gpu 0 --arch resnet34_in --w_lr 0.1001 --mask_epochs 0 --epochs 400 --dataset cifar10\
#  --optim cosine --act_type nn.ReLU --precision half > ./out/resnet34_cifar10_baseline.txt &
# nohup python -u train_cifar.py --gpu 1 --arch resnet34_in --w_lr 0.1001 --mask_epochs 0 --epochs 400 --dataset cifar100\
#  --optim cosine --act_type nn.ReLU --precision half > ./out/resnet34_cifar100_baseline.txt &

# nohup python -u train_cifar.py --gpu 0 --arch resnet18_s --w_lr 0.1 --mask_epochs 0 --epochs 400 --dataset cifar10\
#  --optim cosine --act_type nn.ReLU --precision half > ./out/resnet18_s_cifar10_baseline.txt &

# nohup python -u train_cifar.py --gpu 1 --arch resnet18_s --w_lr 0.1001 --mask_epochs 0 --epochs 400 --dataset cifar100\
#  --optim cosine --act_type nn.ReLU --precision half > ./out/resnet18_s_cifar100_baseline.txt &