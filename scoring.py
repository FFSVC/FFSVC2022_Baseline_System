#! /usr/bin/env python3
import os, sys, time, random, numpy as np
import argparse
import torch, torch.nn as nn, torchaudio, model.resnet as model_2d, model.tdnn as model_1d, model.classifier as classifiers
from torch.utils.data import DataLoader
from dataset import WavDataset
from tools.utils import get_lr, compute_eer
import torch.nn.functional as F
from config.config_scoring import Config
from torch.utils.data import DataLoader
from scipy import spatial

parser = argparse.ArgumentParser(description='Network Parser')
parser.add_argument('--epoch', default=-1, type=int) 
args = parser.parse_args()

def main():
    opt = Config()
    if opt.onlyscoring:
        embd_dict = np.load('exp/%s/%s_%s.npy' % (opt.save_dir, opt.save_name, args.epoch),allow_pickle=True).item()
        eer,_, cost,_ = get_eer(embd_dict, trial_file='data/%s/trials' % opt.val_dir)
        print('Epoch %d\t  EER %.4f\t  cost %.4f\n' % ( args.epoch, eer*100, cost))
        
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

        # validation dataset
        val_dataset = WavDataset(opt=opt, train_mode=False)
        val_dataloader = DataLoader(val_dataset,
                                     num_workers=opt.workers,
                                     batch_size=1,
                                     pin_memory=True)

        if opt.conv_type == '1D':
            model = getattr(model_1d, opt.model)( in_dim=opt.in_planes, embedding_size=opt.embd_dim, hidden_dim=opt.hidden_dim).cuda() # tdnn, ecapa_tdnn
        elif opt.conv_type == '2D':
            model = getattr(model_2d, opt.model)( in_planes=opt.in_planes, embedding_size=opt.embd_dim).cuda()  # resnet


        print('Load exp/%s/model_%d.pkl' % (opt.save_dir, args.epoch))
        checkpoint = torch.load('exp/%s/model_%d.pkl' % (opt.save_dir,  args.epoch))
        model.load_state_dict(checkpoint['model'])
        model = nn.DataParallel(model)
        eer, cost = validate(model,val_dataloader, args.epoch,opt)
        print('Epoch %d\t  EER %.4f\t  cost %.4f\n' % ( args.epoch, eer*100, cost))
  
def get_eer(embd_dict, trial_file):
    true_score = []
    false_score = []

    with open(trial_file) as fh:
        for line in fh:
            line = line.strip()
            utt1, utt2, key = line.split()
            result = 1 - spatial.distance.cosine(embd_dict[utt1], embd_dict[utt2])
            if key == '1':
                true_score.append(result)
            elif key == '0':
                false_score.append(result)  
    eer, threshold, mindct, threashold_dct = compute_eer(np.array(true_score), np.array(false_score))
    return eer, threshold, mindct, threashold_dct

def validate(model,val_dataloader,epoch,opt):
    model.eval()
    embd_dict={}
    with torch.no_grad():
        for j, (feat, utt) in enumerate(val_dataloader):
            outputs = model(feat.cuda())  
            for i in range(len(utt)):
                print(j, utt[i],feat.shape[2])
                embd_dict[utt[i]] = outputs[i,:].cpu().numpy()
    np.save('exp/%s/%s_%s.npy' % (opt.save_dir,opt.save_name, epoch),embd_dict)
    if opt.scoring:
        eer,_, cost,_ = get_eer(embd_dict, trial_file='data/%s/trials' % opt.val_dir)
    else:
        eer, cost = 1,1
    
    return eer, cost


if __name__ == '__main__':
    
    main()
    
