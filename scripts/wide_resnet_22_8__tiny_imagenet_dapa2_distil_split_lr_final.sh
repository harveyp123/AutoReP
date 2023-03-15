#### Train a ReLU pruned model with trainable mask, set the :

### increase lambda to speed up the replacement, otherwise it might not converge to high sparsity
### The lambda also controls the velocity of replacement, the middile velocity may lead to higher accuracy
mkdir -p ./out/


############### Best Accuracy Setting: ############### 
############ For higher ReLU count (higher or equal than 25), better to use lr = 0.001 and 200 epochs
############ For lower ReLU count (lower or equal than 15), better to decrease the learning rate, or number of epochs to prevent overfit
############ Low ReLU count comes with huge model redudancy, and can easily lead to overfit, so a good LR/epoch combination is important


################ Setting 2: Less epochs = 80, high scale_x2 and high clip_x2 ################

########### Accuracy:  ###########
nohup python -u train_tiny_imagenet.py --gpu 6 --arch wide_resnet_22_8 --ReLU_count 150 --w_mask_lr 0.001 --w_lr 0.0001\
    --degree 2  --scale_x2 2 --clip_x2_bool --clip_x2 1 --enable_lookahead \
 --mask_epochs 80 --epochs 0 --lamda 24e1 --batch_size 128 --precision full --dataset tiny_imagenet --data_path /data/tiny-imagenet-200\
 --pretrained_path ./train_cifar/wide_resnet_22_8__tiny_imagenet/cosine_baseline_ReLUs0lr0.1ep400_baseline/best.pth.tar\
 --distil --teacher_arch wide_resnet_22_8 --teacher_path ./train_cifar/wide_resnet_22_8__tiny_imagenet/cosine_baseline_ReLUs0lr0.1ep400_baseline/best.pth.tar\
 --optim cosine --act_type ReLU_masked_dapa_relay --threshold 0.003 > ./out/wide_resnet_22_8_mask_train_dapa2_cnt150_distil.txt &
### Location of checkpoint file, log file, and tb for tensorboard file are in folder: 
### 

