# FFSVC2022 Baseline System

# 0. Introduction

This repository is the FFSVC2022 baseline system, including:

* Data preparation
* Model training
* Embedding extracting
* Performance calculating

Please visit https://ffsvc.github.io/ for more information about the challenge.

# 1. System introduction

The system adopts the online data augmentation method for model training. Please prepare the <a href="https://www.openslr.org/17/">MUSAN </a> and <a href="https://www.openslr.org/17/">RIR_NOISES </a>  dataset and modify the path of './data/MUSAN/' and './data/RIR_Noise/' files as your data path. The acoustic feature extraction and data augmentation depend on the torchaudio package, please make sure your torchaudio version = 0.8.0

Dependencies
```shell
conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=10.2 -c pytorch
```

## Training mode 
The training config is saved in "./config/*.fig" files, and the training log is saved in "exp/PATH_SAVE_DIR".

### DataParallel(DP) Training
```shell
python train.py &
``` 
### DistributedDataParallel(DDP) Training 
```shell
CUDA_VISIBLE_DEVICES="0,1" nohup python -m torch.distributed.launch --nproc_per_node=2 train_dist.py > vox2_resnet.out 2>&1
```

## Test mode
There are three modes for scoring.py,
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

* Compute EER and mDCF

```python
scoring = False/True
onlyscoring = True
``` 

Please set the test mode in the './exp/config_scoring.py' before running the scoring.py

```shell
python scoring.py --epoch 37 &
```

## Pretrained model
We provided the ResNet34-C32 and ECAPA-TDNN-C1024 pre-trained models for participants.
The following are the pre-trained model results on the Vox-O.

|  Model  | Vox-O (EER)  | Download Link |
|  ----  | ----  | ---- |
| ResNet34-C32  | 2.07% | <a href="https://drive.google.com/file/d/1jORY48FfRt7CWWgAtsxJRKd_TjQEBxRO/view?usp=sharing">Google Drive Link </a> |
| ECAPA-TDNN-C1024  | 1.10% | <a href="https://drive.google.com/file/d/1fDqcaKfxMm_DpyvyXy8Nya3KUagmsJj9/view?usp=sharing">Google Drive Link </a>  | 

# 2. FFSVC2022 System Pipeline

For task1,the system adopts the pre-train + finetuning strategy. First, the Vox2dev data is used to train the pre-trained model. Then, Vox2dev and FFSVC2020 data are integrated to finetuning the pre-trained model.

Data preparation -> Training Close-talking model  (with Vox2dev data) -> Far-field model training (finetuning with Vox2dev and FFSVC2020 data)

For task2, the system adopts the pre-train + clustering to generate pseudo label + finetuning strategy. The pre-train step is the same as the task1. Then, all speaker embeddings from the FFSVC20 dataset are extracted using the pre-trained speaker model. We generate the pseudo labels adopting the KMeans algorithm for clustering and "elbow" method for determining the cluster number. Finally, Vox2dev with ground truth and FFSVC2020 data with pseudo label are integrated to finetuning the pre-train model. 

Data preparation -> Training Close-talking model  (with Vox2dev data) -> Extract embeddings of FFSVC2020 data -> Annotate the pseudo label using KMeans algorithm -> Far-field model training (finetuning with Vox2dev and FFSVC2020 data)

The results are 6.7% and 7.2% EER in the task1 and task2 dev set, receptively. 

## Task1

### Step 1. Data preparation
The data preparation file follows the Kaldi form that participants need "wav.scp", "utt2spk" and "spk2utt" files for training dir, and "wav.scp" and "trials" for valuation dir.
The "./data/Vox2dev/" shows the training example files and "./data/Vox1-O" shows the valuation example files. There are five data dir need to be prepared in the baseline system recipe:

```shell
./data/Vox2dev/
    ./wav.scp
    ./utt2spk
    ./spk2utt
./data/Vox1-O/
    ./wav.scp
    ./trials
./data/FFSVC2020_supplement/
    ./wav.scp
    ./utt2spk
    ./spk2utt
./data/FFSVC2022/dev/
    ./wav.scp
    ./trials # with keys, download from "https://ffsvc.github.io/assets/ffsvc2022/trials_dev_keys"
./data/FFSVC2022/eval/ 
    ./wav.scp
    ./trials # without keys, download from "https://ffsvc.github.io/assets/ffsvc2022/trials_eval"
./data/FFSVC2022/Vox2dev_FFSVC22/ # The combination of Vox2dev and FFSVC2020_supplement
    ./wav.scp
    ./utt2spk
    ./spk2utt
```

### Step 2. Training Close-talking model  (training with Vox2dev data)
Modify the parameters in './config/config_resnet_dist.py' or './config/config_resnet.py' before training. The default model is resnet. If you have download the pre-trained model already, please ignore the step.

```shell
python train.py & # training with DP
``` 
or 

```shell
CUDA_VISIBLE_DEVICES="0,1" nohup python -m torch.distributed.launch --nproc_per_node=2 train_dist.py > vox2_resnet34.out 2>&1 & # training with DDP
``` 

### Step 3. Training Far-field model  (finetuning with Vox2dev and FFSVC2020 data)

Modify the training dir as "./data/FFSVC2022/Vox2dev_FFSVC22/" and valuation dir as "./data/FFSVC2022/dev/" before finetuning. 

(if you have download pre-trained model, please put the model into the "save_dir" (in config file) and change the "start_epoch" as 38 (resnet pre-trained model)  the and running：

```shell
python train.py &
``` 
or 

```shell
CUDA_VISIBLE_DEVICES="0,1" nohup python -m torch.distributed.launch --nproc_per_node=2 train_dist.py > vox2_resnet34_ft.out 2>&1 &
```
Note that: 
Since the development set of FFSVC2022 contains about 68,543 audios, it is strongly recommended to comment on the validation code in "train.py" or "train_dist.py". 

### Step 4. Valuation model

Modify './config/config_scoring.py' as the following content,

```python
val_dir = './data/PATH_FFSVC2022/dev'
save_name = 'dev'
scoring = True
onlyscoring = False
```

and running with corresponding model number
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

### Step 4. Annotate the pseudo lebels using KMeans algorithm
Visit KMeans_for_task2.ipynb

### Step 5. Far-field model training (finetuning with Vox2dev and FFSVC2020 data)
After Step 4., concatenate the FFSVC2020 supplement set (with pseudo labels) and Vox2dev dat (with ground truth)
```shell
cd /PATH/Baseline_System/
mkdir ./data/Vox2dev_FFSVC20sup_task2/
cat ./data/FFSVC2022/supplementary/wav.scp ./data/Vox2dev/wav.scp > ./data/Vox2dev_FFSVC20sup_task2/wav.scp
cat ./data/FFSVC2022/supplementary/round1_utt2spk_c100 ./data/Vox2dev/utt2spk > ./data/Vox2dev_FFSVC20sup_task2/utt2spk
cd ./data/Vox2dev_FFSVC20sup_task2/ & ../../tools/utt2spk_to_spk2utt.pl <utt2spk > spk2utt
```

then modify the  ".config.config_resnet_ft_task2" as Config in "train.py", and change the other config parameters. Then running,

```shell
python train.py &
``` 

Repeat Step 4. and Step 5. until the performance is stable on the FFSVC2022 development set. 

# References
Code

https://github.com/ronghuaiyang/arcface-pytorch

https://github.com/ilyaraz/pytorch_kmeans


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
