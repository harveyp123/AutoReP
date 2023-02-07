# ReLU Reduction/ ReLU Peplacement


Some steps to setup the environment 
```bash
# Create a environment
conda create –name torchenv
#or
conda create --prefix=${HOME}/.conda/envs/torchenv python=3.9
# Then activate the environment
conda activate torchenv
# Install pytorch package
conda install -y pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.6 -c pytorch -c conda-forge
# Install tensorboard to record accuracy/loss stuffs
conda install -c conda-forge tensorboardx
pip install tqdm pytorch_warmup
```
The working folder can be found in "```/data3/hop20001/MPC_sparse_act/ACT_prune_cleaned_v6_relu_count```" on ```137.99.3.174```, the folder is set to be visible for any user, so you can load pretrained models or look at results from the working folder. 

## 1. Train a baseline models
Ways to repeat the pretrained model experiment:
```bash
bash scripts/scripts_baseline.sh
```
- You should specify "```--act_type nn.ReLU```" to run the baseline model by using ReLU non-linear function. 
- You can speicify which gpu you will be used by changing "```--gpu 0```". In the scripts, "```nohup python > out.log```" put the execution of python program into background, and direct the command line output to out.log. <br /> 

- The model architecture is specified through "```--arch resnet18```", and the basic architectures can be found in folder locaded in "```models_cifar```". "```resnet18```" architecture is located in "```models_cifar/resnet_basic.py```", which is a smaller version. "```ResNet18```" architecture is located in "```models_cifar/resnet.py```", which is a larger version. 

- For cross-work comparison, Fig. 4 of "Selective Network Linearization for Efficient Private Inference" (denoted as ```SNL```) has the ReLU count for their version of ResNet-18 on CIFAR-10, and it seems that the model is different from both of our "```resnet18```" and "```ResNet18```" model version, the size is in the middle of "```resnet18```" and "```ResNet18```" model. 
- The checkpoint and log file location folder: <br />
"```train_cifar/resnet9__cifar10/cosine_baseline_sp0.0lr0.1ep400_baseline```" (ACC ``` 92.77%```) <br />
"```train_cifar/resnet18__cifar10/cosine_baseline_sp0.0lr0.1ep400_baseline```" (ACC ``` 93.84%```) <br />
"```train_cifar/ResNet18__cifar10/cosine_baseline_sp0.0lr0.1ep400_baseline```" (ACC ``` 95.75%```) <br />

## Add Hysteresis Loop into Gated Mask function:
The experiment shares similar settings as [Sec. 2](#2-run-purely-relu-pruning-optional). However, the forward part of gated mask is a hysteresis loop function rather than simple gated mask as ```f(x) = x > 0```. The backward follows the same STE function as gated mask backward. 

The hysteresis function looks like this: 

![Alt text](figure/Hysteresis.svg)

The hysteresis function can be described as:
```python
def Hysteresis(now_state, in_val, threshold):
    if now_state == 1:
        if in_val < (-1) * threshold:
            now_state = 0
    else:
        if in_val > threshold:
            now_state = 1
    return now_state
```
The threshold is a hyper-parameter to adjust the size of hysteresis loop. 

 

## 2. Run ReLU replacement (pruning) with Polynomial function (w<sub>2</sub>x<sup>2</sup> + w<sub>1</sub>x + w<sub>0</sub>) or (w<sub>1</sub>x + w<sub>0</sub>) with hysteresis loop:

Here are the steps to run the ReLU replacement (pruning) with proposed function: 
- Step 1: We run the experiment for "```resnet18```" architecture: 
    ```bash
    bash scripts/scripts_resnet18_autopoly1.sh
    bash scripts/scripts_resnet18_autopoly1_freezex.sh
    bash scripts/scripts_resnet18_autopoly2.sh
    ```
    - "```scripts_resnet18_autopoly1.sh```" runs the polynomial replacement with (w<sub>1</sub>x + w<sub>0</sub>) function, where w<sub>1</sub> and w<sub>0</sub> are for channel-wise trainable activation function. w<sub>1</sub>, w<sub>0</sub> are initialized as ```1, 0```.
    - "```scripts_resnet18_autopoly1_freezex.sh```" runs the polynomial replacement with (w<sub>1</sub>x + w<sub>0</sub>) function, where w<sub>1</sub> and w<sub>0</sub> are initialized as ```1, 0```, and are not trainable, which is the same function as SNL paper baseline. 
        - We use ```--freezeact``` to freeze the parameter of activation function
    - "```scripts_resnet18_autopoly2.sh```" runs the polynomial replacement with (w<sub>2</sub>x<sup>2</sup> + w<sub>1</sub>x + w<sub>0</sub>) function, where w<sub>2</sub> w<sub>1</sub> and w<sub>0</sub> are for channel-wise trainable activation function. w<sub>2</sub>, w<sub>1</sub>, w<sub>0</sub> are initialized as ```0, 1, 0```.
        - A scaling factor ```--scale_x2 0.2``` is used in the training, to stabilize the w<sub>2</sub>x<sup>2</sup> during the training (reducing w<sub>2</sub> learning rate). If you find the training unstable for larger model, you can decrease ```--scale_x2``` value. 
    - General:
        - "```--ReLU_count```" is used to adjust the number of ```kilo``` ReLU counts we want to achieve, ```--ReLU_count 1.8``` means 1.8K ReLUs. 
        - We change "```--lamda```" parameter to adjust the replacement speed, the higher the ```lambda```, the higher the replacement speed. For less ReLU counts model ,for example is the case of "```--ReLU_count 1.8```", we use "```--lamda 4.0e1```" to ensure we prune such many ReLUs. 
        - We use ```--threshold 0.003``` for the hysteresis threshold, the higher the threshold, the better stability but less exploration. The smaller threshold, the less stability but more exploration. 0.003 is a good balance between stability and exploration. However, more ablation study can be done based on this. 

    - The checkpoint and log file location folder of "```scripts_resnet18_autopoly1.sh```": <br />
"```train_cifar_autopoly1_relay/resnet18__cifar10_relay_0.003/cosine_ReLUs1.8wm_lr0.01mep50_baseline```" (Best: ReLU counts: ```1,800```, ACC ```87.27%```) <br /> 
"```train_cifar_autopoly1_relay/resnet18__cifar10_relay_0.003/cosine_ReLUs12.9wm_lr0.01mep50_baseline```" (Best: ReLU counts: ```12,900```, ACC ```91.44%```) <br /> 
"```train_cifar_autopoly1_relay/resnet18__cifar10_relay_0.003/cosine_ReLUs24.9wm_lr0.01mep50_baseline```" (Best: ReLU counts: ```24,898```, ACC ``` 92.60%```) <br /> 
"```train_cifar_autopoly1_relay/resnet18__cifar10_relay_0.003/cosine_ReLUs51.2wm_lr0.01mep50_baseline```" (Best: ReLU counts: ```51,187```, ACC ``` 92.82%```) <br /> 
    - The checkpoint and log file location folder of "```scripts_resnet18_autopoly1_freezex.sh```": <br />
"```train_cifar_autopoly1_relay_x/resnet18__cifar10_relay_0.003/cosine_ReLUs1.8wm_lr0.01mep50_baseline```" (Best: ReLU counts: ```1,799```, ACC ```85.90%```) <br /> 
"```train_cifar_autopoly1_relay_x/resnet18__cifar10_relay_0.003/cosine_ReLUs12.9wm_lr0.01mep50_baseline```" (Best: ReLU counts: ```12,897```, ACC ```90.56%```) <br /> 
"```train_cifar_autopoly1_relay_x/resnet18__cifar10_relay_0.003/cosine_ReLUs24.9wm_lr0.01mep50_baseline```" (Best: ReLU counts: ```24,893```, ACC ```91.77%```) <br /> 
"```train_cifar_autopoly1_relay_x/resnet18__cifar10_relay_0.003/cosine_ReLUs51.2wm_lr0.01mep50_baseline```" (Best: ReLU counts: ```51,191```, ACC ```92.48%```) <br /> 

    - The checkpoint and log file location folder of "```scripts_resnet18_autopoly2.sh```": <br />
"```train_cifar_autopoly2_relay/resnet18__cifar10_relay_0.003/cosine_ReLUs1.8wm_lr0.01mep50_baseline```" (Best: ReLU counts: ```1,800```, ACC ```90.58%```) <br /> 
"```train_cifar_autopoly2_relay/resnet18__cifar10_relay_0.003/cosine_ReLUs12.9wm_lr0.01mep50_baseline```" (Best: ReLU counts: ```12,900```, ACC ```91.94%```) <br /> 
"```train_cifar_autopoly2_relay/resnet18__cifar10_relay_0.003/cosine_ReLUs24.9wm_lr0.01mep50_baseline```" (Best: ReLU counts: ```24,894```, ACC ```92.21%```) <br /> 
"```train_cifar_autopoly2_relay/resnet18__cifar10_relay_0.003/cosine_ReLUs51.2wm_lr0.01mep50_baseline```" (Best: ReLU counts: ```51,195```, ACC ```92.77%```) <br /> 
- Step 2: We run the finetuning for "```resnet18```" architecture: 
    ```bash
    bash scripts/scripts_resnet18_autopoly1_finetune.sh
    bash scripts/scripts_resnet18_autopoly1_freezex_finetune.sh
    bash scripts/scripts_resnet18_autopoly2_finetune.sh
    ```
    - General:
        - We use the checkpoint from step 1 for finetuning, and fixing the mask. 
    - The checkpoint and log file location folder of "```scripts_resnet18_autopoly1_finetune.sh```": <br />
"```train_cifar_autopoly1_relay/resnet18__cifar10_relay_0.003/cosine_ReLUs1.8lr0.001ep30_baseline```" (Best: ReLU counts: ```1,800```, ACC ```87.49%```) <br /> 
"```train_cifar_autopoly1_relay/resnet18__cifar10_relay_0.003/cosine_ReLUs12.9lr0.001ep30_baseline```" (Best: ReLU counts: ```12,900```, ACC ```91.58%```) <br /> 
"```train_cifar_autopoly1_relay/resnet18__cifar10_relay_0.003/cosine_ReLUs24.9lr0.001ep30_baseline```" (Best: ReLU counts: ```24,898```, ACC ``` 92.54%```) <br /> 
"```train_cifar_autopoly1_relay/resnet18__cifar10_relay_0.003/cosine_ReLUs51.2lr0.001ep30_baseline```" (Best: ReLU counts: ```51,187```, ACC ``` 92.86%```) <br /> 
    - The checkpoint and log file location folder of "```scripts_resnet18_autopoly1_freezex_finetune.sh```": <br />
"```train_cifar_autopoly1_relay_x/resnet18__cifar10_relay_0.003/cosine_ReLUs1.8lr0.001ep30_baseline```" (Best: ReLU counts: ```1,799```, ACC ```86.31%```) <br /> 
"```train_cifar_autopoly1_relay_x/resnet18__cifar10_relay_0.003/cosine_ReLUs12.9lr0.001ep30_baseline```" (Best: ReLU counts: ```12,897```, ACC ```90.83%```) <br /> 
"```train_cifar_autopoly1_relay_x/resnet18__cifar10_relay_0.003/cosine_ReLUs24.9lr0.001ep30_baseline```" (Best: ReLU counts: ```24,893```, ACC ```92.04%```) <br /> 
"```train_cifar_autopoly1_relay_x/resnet18__cifar10_relay_0.003/cosine_ReLUs51.2lr0.001ep30_baseline```" (Best: ReLU counts: ```51,191```, ACC ```92.66%```) <br /> 

    - The checkpoint and log file location folder of "```scripts_resnet18_autopoly2_finetune.sh```": <br />
"```train_cifar_autopoly2_relay/resnet18__cifar10_relay_0.003/cosine_ReLUs1.8lr0.001ep30_baseline```" (Best: ReLU counts: ```1,800```, ACC ```90.81%```) <br /> 
"```train_cifar_autopoly2_relay/resnet18__cifar10_relay_0.003/cosine_ReLUs12.9lr0.001ep30_baseline```" (Best: ReLU counts: ```12,900```, ACC ```91.93%```) <br /> 
"```train_cifar_autopoly2_relay/resnet18__cifar10_relay_0.003/cosine_ReLUs24.9lr0.001ep30_baseline```" (Best: ReLU counts: ```24,894```, ACC ```92.36%```) <br /> 
"```train_cifar_autopoly2_relay/resnet18__cifar10_relay_0.003/cosine_ReLUs51.2lr0.001ep30_baseline```" (Best: ReLU counts: ```51,195```, ACC ```92.94%```) <br /> 
- Step 3 (Optional): We run the evaluation for "```resnet18```" architecture:
    ```bash
    bash scripts/scripts_resnet18_autopoly1_evaluate.sh
    bash scripts/scripts_resnet18_autopoly1_finetune_evaluate.sh
    ```
    - Results located in ```evaluate_cifar_autopoly1_relay```. 
