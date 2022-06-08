# FFSVC2022 Baseline System

# 0. Introduction

This repository is the FFSVC2022 baseline system, including:

* Data preparation
* Model training
* Embedding extracting
* Performance calculating

Please visit https://github.io/ffsvc.io for more information about the challege.

# 1. System introduction

The system adopts the online data augmentation method for model training. Please prepare the <a href="https://www.openslr.org/17/">MUSAN </a> and <a href="https://www.openslr.org/17/">RIR_NOISES </a>  dataset and modify the path of './data/musan' and './data/rir_noise/' files as your saved path. The acoustic feature depends on the torchaudio package, please make sure your torchaudio version > 0.8.0

## Training mode 
The training config saved in "./config/*.fig" files and the training log saved in "exp/PATH_SAVE_DIR".

### DataParallel(DP) Trianing
```shell
python trian.py &
``` 
### DistributedDataParallel(DDP) Training 
```shell
CUDA_VISIBLE_DEVICES="0,1" nohup python -m torch.distributed.launch --nproc_per_node=2 train_dist.py > vox2_resnet.out 2>&1
```

## Test mode
There are three mode for scoring.py,
* Extract speaker embedding and compute the EER and mDCF 
```python
scoring = True
onlyscoring = False
``` 

* Extract speaker embedding
```python
scoring = False
onlyscoring = False
``` 

* Compute the EER and mDCF

```python
scoring = False/True
onlyscoring = True
``` 
Please setting the test mode in the './exp/config_scoring.py' before running the scoring.py

```shell
python scoring.py --epoch 1 &
```

## Pretrained model
We provided the ResNet34-C32 and ECAPA-TDNN-C1024 pretrained model for participants.
The following is the pretrained model results on the Vox-O

|  Model  | Vox-O (EER)  | Download Link |
|  ----  | ----  | ---- |
| ResNet34-C32  | 2.07% | <a href="https://drive.google.com/file/d/1jORY48FfRt7CWWgAtsxJRKd_TjQEBxRO/view?usp=sharing">Google Drive Link </a> |
| ECAPA-TDNN-C1024  | 1.10% | <a href="https://drive.google.com/file/d/1fDqcaKfxMm_DpyvyXy8Nya3KUagmsJj9/view?usp=sharing">Google Drive Link </a>  | 

# 2. FFSVC2022 System Pipepline

For task1:
Data preparation -> Training Close-talking model  (with Vox2dev data) -> Far-field model training (finetuning with Vox2dev and FFSVC2020 data)

For task2：
Data preparation -> Training Close-talking model  (with Vox2dev data) -> Extract embeddings of FFSVC2020 data -> Annotate the pseudo lebel using KMeans algorithm -> Far-field model training (finetuning with Vox2dev and FFSVC2020 data)

## Task1

### Step 1. Data preparation
The data preparation file follows the Kaldi form that participants need "wav.scp", "utt2spk" and "spk2utt" files for training dir, and "wav.scp" and "trials" for valuation dir.
The "./data/Vox2dev/" shows the training example fiels and "./data/Vox1-O" shows the valuation example files. There are five data dir need to be prepared:

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
    ./trials # with keys, download from <a href="https://ffsvc.github.io/assets/ffsvc2022/trials_dev_keys"> FFSVC22 dev trails</a> 
./data/FFSVC2022/eval/ 
    ./wav.scp
    ./trials # withour keys, download from <a href="https://ffsvc.github.io/assets/ffsvc2022/trials_eval"> FFSVC22 eval trails</a> 
./data/FFSVC2022/Vox2dev_FFSVC22/ # The combination of Vox2dev and FFSVC2020_supplement
    ./wav.scp
    ./utt2spk
    ./spk2utt
```

### Step 2. Training Close-talking model  (with Vox2dev data)
Modify the parameters in './config/config_resnet_dist.py' or './config/config_resnet.py'. The defalut model is resnet. 

```shell
python trian.py & # training with DP
``` 
or 

```shell
CUDA_VISIBLE_DEVICES="0,1" nohup python -m torch.distributed.launch --nproc_per_node=2 train_dist.py > vox2_resnet34.out 2>&1 & # training with DDP
``` 

### Step 3. Training Far-field model  (finetuning with Vox2dev and FFSVC2020 data)

Modify the training dir as "./data/FFSVC2022/Vox2dev_FFSVC22/" and valuation dir as "./data/FFSVC2022/dev/"

```shell
python trian.py &
``` 
or 

```shell
CUDA_VISIBLE_DEVICES="0,1" nohup python -m torch.distributed.launch --nproc_per_node=2 train_dist.py > vox2_resnet34_ft.out 2>&1 &
```
Note that: 
The development set of FFSVC2022 contain about 68,543 audios, it is strongly recommended to comment the validation code in "train.py". 

### Step 4. Valuation model

Modify './config/config_scoring.py' as the following content,
```python
val_dir = './data/PATH_FFSVC2022/dev'
save_name = 'dev'
scoring = True
onlyscoring = False
```

and running

```shell
python scoring.py --epoch 37 &
```

## Task2

### Step 1. Data preparation

Same as task1.

### Step 2. Training Close-talking model  (with Vox2dev data)

Same as task1.

### Step 3. Extract embeddings of FFSVC2020 data
Modify './config/config_scoring.py' as the following content,
```python
val_dir = './data/PATH_FFSVC2022_supplementary'
save_name = 'supplementary'
scoring = False
onlyscoring = False
```

and running 
```shell
python scoring.py --epoch 37 &
```

### Step 4. Annotate the pseudo lebel using KMeans algorithm
Visit KMeans_for_task2.ipynb

### Step 5. Far-field model training (finetuning with Vox2dev and FFSVC2020 data)
After Step 4., concatenate the FFSVC2020 supplement set (with pseudo labels) and Vox2dev dat (with ground truth)
```shell
cd /PATH/Baseline_System/
mkdir ./data/Vox2dev_FFSVC20sup_task2/
cat ./data/FFSVC2022/supplementary/wav.scp ./data/Vox2dev/wav.scp > ./data/Vox2dev_FFSVC20sup_task2/wav.scp
cat ./data/FFSVC2022/supplementary/round1_utt2spk_c100 ./data/Vox2dev/utt2spk > ./data/Vox2dev_FFSVC20sup_task2/utt2spk
cd ./data/Vox2dev_FFSVC20sup_task2/ &../../tools/utt2spk_to_spk2utt.pl <utt2spk > spk2utt
```

then modify the ".config.config_resnet_ft_task2" as Config in "train.py" 
training dir as "./data/Vox2dev_FFSVC20sup_task2/" and running

```shell
python trian.py &
``` 

Repeat Step 4. and Step 5. Until the performance is stable on the FFSVC2022 development set. 

# References
Code

https://github.com/ronghuaiyang/arcface-pytorch


Papers

[1] On-the-Fly Data Loader and Utterance-Level Aggregation for Speaker and Language Recognition
```shell
@ARTICLE{9036861,
  author={Cai, Weicheng and Chen, Jinkun and Zhang, Jun and Li, Ming},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing}, 
  title={On-the-Fly Data Loader and Utterance-Level Aggregation for Speaker and Language Recognition}, 
  year={2020},
  volume={28},
  number={},
  pages={1038-1051},
  doi={10.1109/TASLP.2020.2980991}}
```
 
[2] An Iterative Framework for Self-Supervised Deep Speaker Representation Learning
```shell
@INPROCEEDINGS{9414713,
  author={Cai, Danwei and Wang, Weiqing and Li, Ming},
  booktitle={ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={An Iterative Framework for Self-Supervised Deep Speaker Representation Learning}, 
  year={2021},
  volume={},
  number={},
  pages={6728-6732},
  doi={10.1109/ICASSP39728.2021.9414713}}
  ```
