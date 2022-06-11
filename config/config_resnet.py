class Config(object):
    save_dir = 'Vox2dev_80FBANK_ResNet34StatsPool_AMsoftmax_256'
    train_dir = 'Vox2dev'
    val_dir = 'Vox1-O'
    
    workers = 10
    batch_size = 128
    max_frames = 200
    
    data_wavaug = True
    data_specaug = True
    specaug_masktime = [10,20]
    specaug_maskfreq = [5,10]
    
    noise_dir = 'MUSAN'
    rir_dir = 'RIR_Noise'
    fs = 16000
    nfft = 512
    win_len = 0.025
    hop_len = 0.01
    n_mels = 80
    
    conv_type = '2D' #1D, 2D
    model = 'ResNet34StatsPool' # ResNet34StatsPool,TDNN,ECAPA_TDNN
    in_planes = 32 # conv_type:1D, in_planes=n_mels; 2D, in_planes=32, 64
    embd_dim = 256
    hidden_dim = 1024
    classifier = 'AAMSoftmax' # AAMSoftmax,AMSoftmax, ASoftmax, Softmax
    angular_m = 0.4
    angular_s = 32
    
    warm_up_epoch = 2
    lr = 0.001

    epochs = 100
    start_epoch = 40
    load_classifier = False
    
    seed = 3007
    gpu = '4,5'
    
    utt2wav = [line.split() for line in open('data/%s/wav.scp' % train_dir)]
    spk2int = {line.split()[0]:i for i, line in enumerate(open('data/%s/spk2utt' % train_dir))}
    spk2utt = {line.split()[0]:line.split()[1:] for line in open('data/%s/spk2utt' % train_dir)}
    
    utt2wav_val = [line.split() for line in open('data/%s/wav.scp' % val_dir)]

    noise_list = {'noise': [i.strip('\n') for i in open('data/%s/noise_wav_list'% noise_dir) ],
              'music': [i.strip('\n') for i in open('data/%s/music_wav_list'% noise_dir) ],
              'reverb': [i.strip('\n') for i in open('data/%s/rir_list'% rir_dir) ]}
