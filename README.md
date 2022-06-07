# FFSVC2022 Baseline System

# Introduction

This repository is the FFSVC2022 baseline system, including:

* Data preparation
* Model training
* Embedding extracting
* Performance calculating

Please visit https://github.io/ffsvc.io for more information about the challege.

# System introduction

The system adopts the online data augmentation method.

## Training mode 
The training config saved in "./config/*.fig" files and the training log saved in "exp/PATH_SAVE_DIR"

### DP Trianing
```shell
python trian.py &
``` 

### DDP Training 
```shell
CUDA_VISIBLE_DEVICES="0,1" nohup python -m torch.distributed.launch --nproc_per_node=2 train_dist.py > vox2_resnet.out 2>&1
```

## Test mode
```shell
python scoring.py --epoch 1 &
```

## Pretrained model
We provided the ResNet34-C32 and ECAPA-TDNN-C1024 pretrained model for user.


# System Pipepline

For task1:
Data preparation -> Training Close-talking model  (with Vox2dev data) -> Far-field model training (finetuning with Vox2dev and FFSVC2020 data)

For task2：
Data preparation -> Training Close-talking model  (with Vox2dev data) -> Extract embeddings of FFSVC2020 data -> Annotate the pseudo lebel using KMeans algorithm -> Far-field model training (finetuning with Vox2dev and FFSVC2020 data)

## Task1

### Step 1. Data preparation
The data preparation file follows the Kaldi form that user needs "wav.scp", "utt2spk" and "spk2utt" for training dir, and "wav.scp" and "trials" for valuation dir
The "./data/Vox2dev/" shows the training example fiels and "./data/Vox1-O" shows the valuation example files.

there are five data dir need to be prepared:

```shell
./data/Vox2dev/
    ./wav.scp
    ./utt2spk
    ./spk2utt
./data/Vox1-O/
    ./wav.scp
    ./utt2spk
    ./spk2utt
./data/FFSVC2020_supplement/
    ./wav.scp
    ./utt2spk
    ./spk2utt
./data/FFSVC2022/dev/
    ./wav.scp
    ./trials # with keys  download from https://
./data/FFSVC2022/eval/ 
    ./wav.scp
    ./trials # withour keys, download from https://
./data/FFSVC2022/Vox2dev_FFSVC22/ # The combination of Vox2dev and FFSVC2020_supplement
    ./wav.scp
    ./utt2spk
    ./spk2utt
```


### Step 2. Training Close-talking model  (with Vox2dev data)

```shell
python trian.py &
``` 
or 

```shell
CUDA_VISIBLE_DEVICES="0,1" nohup python -m torch.distributed.launch --nproc_per_node=2 train_dist.py > vox2_resnet34.out 2>&1 &
``` 

### Step 3. Training Far-field model  (finetuning with Vox2dev and FFSVC2020 data)

modify the training dir as "./data/FFSVC2022/Vox2dev_FFSVC22/" and valuation dir as "./data//FFSVC2022/dev/"

```shell
python trian.py &
``` 
or 

```shell
CUDA_VISIBLE_DEVICES="0,1" nohup python -m torch.distributed.launch --nproc_per_node=2 train_dist.py > vox2_resnet34_ft.out 2>&1 &
```
Note that: 
The development set of FFSVC2022 contain about 80,000 audio, it is strongly recommended to annotate the test code in "train.py" file 

### Step 4. Valuation model

```shell
python scoring.py --epoch 1 &
```

## Task2

### Step 1. Data preparation

Same as task1.

### Step 2. Training Close-talking model  (with Vox2dev data)

Same as task1.

### Step 3. Extract embeddings of FFSVC2020 data

```shell
python scoring.py --epoch 1 &
```

### Step 4. Annotate the pseudo lebel using KMeans algorithm
Visit KMeans.ipynb

### Step 5. Far-field model training (finetuning with Vox2dev and FFSVC2020 data)
modify the training dir as "./data/FFSVC2022/Vox2dev_FFSVC22_task2/" and valuation dir as "./data//FFSVC2022/dev/"
```shell
python trian.py &
``` 

Repeat Step 4. and Step 5. until the performance stable on the FFSVC2022 development set. 


### References
Coda 

Papers
[1] W. Cai online data aug
[2] D. Cai kmeans


