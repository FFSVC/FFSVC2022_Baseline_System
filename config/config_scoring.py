class Config(object):
    save_dir = 'Vox2dev_80FBANK_ResNet34StatsPool_AAMsoftmax_256'
    val_dir = 'FFSVC2022/supplementary'
    save_name = 'supplementary'
    scoring = False     # True: extract and scoring 
                   # False: extract, not scoring
    onlyscoring = False  # True : has npy
                   # False : no npy

    workers = 10
    batch_size = 1
    max_frames=200
    
    fs = 16000
    nfft = 512
    win_len = 0.025
    hop_len = 0.01
    n_mels = 80
    
    conv_type = '2D' #1D, 2D
    model = 'ResNet34StatsPool' # ResNet34StatsPool,TDNN,ECAPA_TDNN
    in_planes = 32 # conv_type:1D, in_planes=n_mels; 2D, in_planes=32/64
    embd_dim = 256
    hidden_dim = 1024 # ECAPA_TDNN

    gpu = '0'
    
    utt2wav_val = [line.split() for line in open('data/%s/wav.scp' % val_dir)]
