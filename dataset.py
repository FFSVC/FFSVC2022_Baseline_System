# dataset
from torch.utils.data import Dataset, DataLoader
import torchaudio
import random
from tools.processing import *
import numpy as np

class WavDataset(Dataset):
    """Wav dataset for Speaker Verification."""

    def __init__(self, opt=None, train_mode=True):
        """
        Args:
            utt2wav (list): 
            utt2spk (dict): 
            opt :
        """
        self.opt = opt
        
        self.train_mode = train_mode
        self.max_frames = opt.max_frames
        self.fs = opt.fs
        
        if train_mode:
            
            self.noise_list = opt.noise_list
            self.utt2wav = opt.utt2wav
            self.spk2int = opt.spk2int
            self.data_wavaug = opt.data_wavaug
            self.data_specaug = opt.data_specaug
            self.specaug_masktime = opt.specaug_masktime
            self.specaug_maskfreq = opt.specaug_maskfreq
            
            self.utt2spk = {line.split()[0]:opt.spk2int[line.split()[1]] for line in open('data/%s/utt2spk' % opt.train_dir)}
            
        else:
            self.utt2wav = opt.utt2wav_val
            
        self.transforms = torchaudio.transforms.MelSpectrogram(sample_rate= opt.fs, 
                                     n_fft = opt.nfft, 
                                     win_length=int(opt.fs*opt.win_len), 
                                     hop_length=int(opt.fs*opt.hop_len), 
                                     n_mels = opt.n_mels)

    def __len__(self):
        return len(self.utt2wav)
    
    def __getitem__(self, idx):
        utt, filename = self.utt2wav[idx]
        signal = load_wav(filename, max_frames=self.max_frames, fs=self.fs, train_mode=self.train_mode)
        if self.train_mode:           
            if self.data_wavaug:
                aug_type = random.sample(['noise', 'rir', 'clean', 'clean'],1)[0]
                if aug_type == 'noise':
                    signal = addnoise(signal,self.noise_list,max_frames=self.max_frames)
                elif aug_type == 'rir':
                    signal = addreverberate(signal,self.noise_list,max_frames=self.max_frames)
                    
            signal = mean_std_norm_1d(signal)
                
            feat = torch.log(self.transforms(signal) + 1e-6)
            feat = feat - feat.mean(axis=1).unsqueeze(1)
            
            if self.data_specaug:
                aug_type = random.sample(['specaug','clean','clean'],1)[0]
                if aug_type == 'specaug':
                    feat = augment_spec(feat,self.opt)
                    
            return feat,self.utt2spk[utt]
        
        else:
            feat = torch.log(self.transforms(mean_std_norm_1d(signal)) + 1e-6)
            feat = feat - feat.mean(axis=1).unsqueeze(1)
#             feat = feat - feat.mean()
            return feat,utt
        