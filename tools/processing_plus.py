import random
import math
import torch
import torchaudio
import torchaudio.transforms as T
from scipy import signal as scisignal
import numpy as np

def addspeed(speech,speed_value='1'):
    effects = [['speed',speed_value],
           ['rate', '16000']]
    signal_sox, sample_rate = torchaudio.sox_effects.apply_effects_tensor(speech.unsqueeze(0), 16000, effects, channels_first=True)
    return signal_sox[0]

def addtempo(speech):
    effects = [['tempo', str(random.choice([0.7,0.8,0.9,1.1,1.2,1.3]))]]
    signal_sox, sample_rate = torchaudio.sox_effects.apply_effects_tensor(speech.unsqueeze(0), 16000, effects, channels_first=True)
    return signal_sox[0]

def addvol(speech):
    effects = [['vol', str(random.random() * 15 + 5)]]
    signal_sox, sample_rate = torchaudio.sox_effects.apply_effects_tensor(speech.unsqueeze(0), 16000, effects, channels_first=True)
    return signal_sox[0]

def addnoise(speech,noise_list,snr_dur=[0,20],max_frames=200):
    noies_type = random.sample(['noise','music'],1)[0]
    noise = truncate_speech(load_wav(random.choice(noise_list[noies_type])),max_frames=max_frames, train_mode=True)
    speech_power = speech.norm(p=2)
    noise_power = noise.norm(p=2)
    snr_db =snr_dur[0]+(snr_dur[1]-snr_dur[0])*random.random()
    snr = math.exp(snr_db / 10)
    scale = snr * noise_power / speech_power
    noisy_speech = (scale * speech + noise) / 2
    return noisy_speech

def addreverberate(speech,noise_list,max_frames):
    rir_raw = load_wav(random.choice(noise_list['reverb']))
    signal_rir = scisignal.convolve(speech, rir_raw, mode='full')[:max_frames * 160-160]
    return torch.tensor(signal_rir)

def _get_sample(path, resample=None):
    effects = [
        ["remix", "1"]
    ]
    if resample:
        effects.extend([
            ["lowpass", f"{resample // 2}"],
            ["rate", f'{resample}'],
        ])
    return torchaudio.sox_effects.apply_effects_file(path, effects=effects)

def load_wav(path, fs=16000):
    signal,fs = _get_sample(path,resample=fs)
    return signal[0]
    
def truncate_speech(signal, max_frames, train_mode=False):
    signalsize = signal.shape[0]
    if train_mode:
        max_audio = max_frames * 160-160
        if signalsize <= max_audio:
            signal = signal.repeat(max_audio//signalsize+1)[:max_audio]
        else:
            startframe = np.array([np.int64(random.random()*(signalsize-max_audio))])[0]
            signal = signal[startframe:startframe+max_audio]
        return signal
    else:
        return signal


def mean_std_norm_1d(signal):
    """signal 1D tensor"""
    
    mean = torch.mean(signal).detach().data
    std = torch.std(signal).detach().data
    std = torch.max(std, torch.tensor(1e-6))
    return (signal-mean)/std

def augment_spec(feat,opt):
    
    if random.sample(['time','freq'],1)[0]== 'time':
        param = random.randint(opt.specaug_masktime[0],opt.specaug_masktime[1])
        masking = T.TimeMasking(time_mask_param=param)
    else:
        param = random.randint(opt.specaug_maskfreq[0],opt.specaug_maskfreq[1])
        masking = T.FrequencyMasking(freq_mask_param=4)
    return masking(feat)
    