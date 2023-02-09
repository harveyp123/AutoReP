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

nohup python -u train_cifar.py --gpu 5 --arch resnet18_in --dataset cifar10 --w_lr 0.1 --mask_epochs 0 --epochs 400 --dataset cifar100\
 --optim cosine --act_type nn.ReLU > ./out/resnet18_in_cifar100_baseline.txt &
