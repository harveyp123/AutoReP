#### Train a ReLU pruned model with trainable mask, set the :

### increase lambda to speed up the replacement, otherwise it might not converge to high sparsity
### The lambda also controls the velocity of replacement, the middile velocity may lead to higher accuracy
mkdir -p ./out/

nohup python -u train_cifar.py --gpu 0 --arch resnet18 --ReLU_count 1.8 --w_mask_lr 0.01 --w_lr 0.001\
 --mask_epochs 1 --epochs 0 --degree 1 \
 --evaluate ./train_cifar_autopoly1_relay/resnet18__cifar10_relay_0.003/cosine_ReLUs1.8wm_lr0.01mep50_baseline/best_mask_train.pth.tar\
 --optim cosine --act_type ReLU_masked_autopoly_relay --threshold 0.003> ./out/resnet18_mask_train_autopoly_cnt1.8_evaluate.txt &

nohup python -u train_cifar.py --gpu 1 --arch resnet18 --ReLU_count 12.9 --w_mask_lr 0.01 --w_lr 0.001\
 --mask_epochs 1 --epochs 0 --degree 1 \
 --evaluate ./train_cifar_autopoly1_relay/resnet18__cifar10_relay_0.003/cosine_ReLUs12.9wm_lr0.01mep50_baseline/best_mask_train.pth.tar\
 --optim cosine --act_type ReLU_masked_autopoly_relay --threshold 0.003> ./out/resnet18_mask_train_autopoly_cnt12.9_evaluate.txt &

nohup python -u train_cifar.py --gpu 2 --arch resnet18 --ReLU_count 24.9 --w_mask_lr 0.01 --w_lr 0.001\
 --mask_epochs 1 --epochs 0 --degree 1 \
 --evaluate ./train_cifar_autopoly1_relay/resnet18__cifar10_relay_0.003/cosine_ReLUs24.9wm_lr0.01mep50_baseline/best_mask_train.pth.tar\
 --optim cosine --act_type ReLU_masked_autopoly_relay --threshold 0.003> ./out/resnet18_mask_train_autopoly_cnt24.9_evaluate.txt &

nohup python -u train_cifar.py --gpu 3 --arch resnet18 --ReLU_count 51.2 --w_mask_lr 0.01 --w_lr 0.001\
 --mask_epochs 1 --epochs 0 --degree 1 \
 --evaluate ./train_cifar_autopoly1_relay/resnet18__cifar10_relay_0.003/cosine_ReLUs51.2wm_lr0.01mep50_baseline/best_mask_train.pth.tar\
 --optim cosine --act_type ReLU_masked_autopoly_relay --threshold 0.003> ./out/resnet18_mask_train_autopoly_cnt51.2_evaluate.txt &


# nohup python -u train_cifar.py --gpu 1 --arch resnet18 --ReLU_count 1.8 --w_mask_lr 0.01 --w_lr 0.001\
#  --mask_epochs 0 --epochs 0 --degree 1 --lamda 4.0e1\
#  --evaluate ./train_cifar_autopoly1_relay/resnet18__cifar10_relay_0.003/cosine_ReLUs1.8lr0.001ep30_baseline/best.pth.tar\
#  --optim cosine --act_type ReLU_masked_autopoly_relay --threshold 0.003> ./out/resnet18_mask_train_autopoly_cnt1.8_finetune_evaluate.txt &

