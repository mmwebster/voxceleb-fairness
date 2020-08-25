#!/usr/bin/python
#-*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import math, pdb, sys, random
import numpy as np
import time, os, itertools, shutil, importlib
from baseline_misc.tuneThreshold import tuneThresholdfromScore
from DatasetLoader import extract_eval_subsets_from_spectrogram
from loss.ge2e import GE2ELoss
from loss.angleproto import AngleProtoLoss
from loss.cosface import AMSoftmax
from loss.arcface import AAMSoftmax
from loss.softmax import SoftmaxLoss
from loss.protoloss import ProtoLoss
from loss.pairwise import PairwiseLoss
import torch.cuda.amp as amp

from utils.misc_utils import print_throttler

import wandb

class SpeakerNet(nn.Module):

    def __init__(self, device, max_frames, lr = 0.0001, margin = 1, scale = 1, hard_rank = 0, hard_prob = 0, model="alexnet50", nOut = 512, nSpeakers = 1000, optimizer = 'adam', encoder_type = 'SAP', normalize = True, trainfunc='contrastive', **kwargs):
        super(SpeakerNet, self).__init__();

        argsdict = {'nOut': nOut, 'encoder_type':encoder_type}

        self.device = device

        # grab actual model version
        SpeakerNetModel = importlib.import_module('models.'+model).__getattribute__(model)

        # set model as __S__ member
        self.__S__ = SpeakerNetModel(**argsdict).to(self.device);

        if trainfunc == 'angleproto':
            self.__L__ = AngleProtoLoss(self.device).to(self.device)
            self.__train_normalize__    = True
            self.__test_normalize__     = True
        elif trainfunc == 'ge2e':
            self.__L__ = GE2ELoss().to(self.device)
            self.__train_normalize__    = True
            self.__test_normalize__     = True
        elif trainfunc == 'amsoftmax':
            self.__L__ = AMSoftmax(in_feats=nOut, n_classes=nSpeakers, m=margin, s=scale).to(self.device)
            self.__train_normalize__    = False
            self.__test_normalize__     = True
        elif trainfunc == 'aamsoftmax':
            self.__L__ = AAMSoftmax(in_feats=nOut, n_classes=nSpeakers, m=margin, s=scale).to(self.device)
            self.__train_normalize__    = False
            self.__test_normalize__     = True
        elif trainfunc == 'softmax':
            self.__L__ = SoftmaxLoss(in_feats=nOut, n_classes=nSpeakers).to(self.device)
            self.__train_normalize__    = False
            self.__test_normalize__     = True
        elif trainfunc == 'proto':
            self.__L__ = ProtoLoss().to(self.device)
            self.__train_normalize__    = False
            self.__test_normalize__     = False
        elif trainfunc == 'triplet':
            self.__L__ = PairwiseLoss(loss_func='triplet', hard_rank=hard_rank, hard_prob=hard_prob, margin=margin).to(self.device)
            self.__train_normalize__    = True
            self.__test_normalize__     = True
        elif trainfunc == 'contrastive':
            self.__L__ = PairwiseLoss(loss_func='contrastive', hard_rank=hard_rank, hard_prob=hard_prob, margin=margin).to(self.device)
            self.__train_normalize__    = True
            self.__test_normalize__     = True
        else:
            raise ValueError('Undefined loss.')

        if optimizer == 'adam':
            self.__optimizer__ = torch.optim.Adam(self.parameters(), lr = lr);
        elif optimizer == 'sgd':
            self.__optimizer__ = torch.optim.SGD(self.parameters(), lr = lr, momentum = 0.9, weight_decay=5e-5);
        else:
            raise ValueError('Undefined optimizer.')

        self.__max_frames__ = max_frames;

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Train network
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def train_network(self, loader):

        self.train();

        stepsize = loader.batch_size;

        counter = 0;
        index   = 0;
        loss    = 0;
        top1    = 0     # EER or accuracy

        criterion = torch.nn.CrossEntropyLoss()
        # mixed precision scaler
        scaler = amp.GradScaler()

        print_interval_percent = 2
        print_interval = 0
        print_interval_start_time = time.time()
        epoch_start_time = time.time()


        for data, data_label in loader:
            # init print interval after data loader has set its length during __iter__
            if print_interval == 0:
                num_batches = (len(loader)/loader.batch_size)
                print_interval = max(int(num_batches*print_interval_percent/100), 1)
                print(f"SpeakerNet: Starting training @ {print_interval_percent}%"
                      f" update interval")

            self.zero_grad();

            feat = []
            # use autocast for half precision where possible
            with amp.autocast():
                # @TODO Can alls inp-s in data go into a single batch to
                #       populate feats?
                for inp in data:
                    outp      = self.__S__.forward(inp.to(self.device))
                    if self.__train_normalize__:
                        outp   = F.normalize(outp, p=2, dim=1)
                    feat.append(outp)

                feat = torch.stack(feat,dim=1).squeeze()

                label   = torch.LongTensor(data_label).to(self.device)

                nloss, prec1 = self.__L__.forward(feat,label)

                loss    += nloss.detach().cpu();
                top1    += prec1
                counter += 1;
                index   += stepsize;

            # run backward pass and step optimizer using the autoscaler
            # to mitigate half-precision convergence issues
            scaler.scale(nloss).backward()
            scaler.step(self.__optimizer__)
            scaler.update()

            if counter % print_interval == 0:
                # not sure how to format in f-format str
                interval_elapsed_time = time.time() - print_interval_start_time
                print_interval_start_time = time.time()
                eer_str = "%2.3f%%"%(top1/counter)
                # misc progress updates and estimates
                progress_percent = int(index * 100 / len(loader))
                num_samples_processed = print_interval * loader.batch_size
                sample_train_rate = num_samples_processed / interval_elapsed_time
                epoch_train_period = (len(loader) / sample_train_rate) / 60
                print(f"SpeakerNet: Processed {progress_percent}% (of {len(loader)}) => "
                      f"Loss {loss/counter:.2f}, "
                      f"EER/T1 {eer_str}, "
                      f"Train-rate {sample_train_rate:.2f} samples/sec "
                      f"(est. {epoch_train_period:.2f} mins/epoch)")

        print(f"SpeakerNet: Finished epoch in {(time.time() - epoch_start_time)/60:.2f} mins")
        return (loss/counter, top1/counter);

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Read data from list
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def readDataFromList(self, listfilename):

        data_list = {};

        with open(listfilename) as listfile:
            while True:
                line = listfile.readline();
                if not line:
                    break;

                data = line.split();
                filename = data[1];
                speaker_name = data[0]

                if not (speaker_name in data_list):
                    data_list[speaker_name] = [];
                data_list[speaker_name].append(filename);

        return data_list


    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Evaluate from list
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def evaluateFromListSave(self, listfilename, print_interval_percent=10, feat_dir='', test_path='', num_eval=10):
        
        self.eval();
        
        lines       = []
        files       = []
        filedict    = {}
        feats       = {}
        tstart      = time.time()

        if feat_dir != '':
            print('Saving temporary files to %s'%feat_dir)
            if not(os.path.exists(feat_dir)):
                os.makedirs(feat_dir)

        ## Read all lines
        with open(listfilename) as listfile:
            while True:
                line = listfile.readline();
                if (not line): #  or (len(all_scores)==1000) 
                    break;

                data = line.split();

                files.append(data[1])
                files.append(data[2])
                lines.append(line)

        setfiles = list(set(files))
        setfiles.sort()

        print_interval = int(len(setfiles)*print_interval_percent/100)
        ## Save all features to file
        for idx, file in enumerate(setfiles):
            # extract Subsets x Freq x Frames tensor from a single utterance in
            # evaluation set for evaluation features
            utterance_file_path = os.path.join(test_path,file).replace(".wav", ".npy")
            full_utterance_spectrogram = np.load(utterance_file_path)

            # evaluate on network with half-precision where possible
            with amp.autocast():
                overlapping_spectrogram_subsets = torch.FloatTensor(
                        extract_eval_subsets_from_spectrogram(
                            full_utterance_spectrogram, self.__max_frames__)).to(self.device)

                ref_feat = self.__S__.forward(
                        overlapping_spectrogram_subsets).detach().cpu()

                filename = '%06d.wav'%idx

                if feat_dir == '':
                    feats[file]     = ref_feat
                else:
                    filedict[file]  = filename
                    torch.save(ref_feat,os.path.join(feat_dir,filename))

                telapsed = time.time() - tstart

                if idx % print_interval == 0:
                    print(f"Reading {idx}/{len(setfiles)}: {(idx/telapsed):.2f} Hz, embed size {ref_feat.size()[1]}")

        all_scores = [];
        all_labels = [];
        tstart = time.time()

        total_length = len(lines)
        print_interval = int(total_length*print_interval_percent/100)
        ## Read files and compute all scores
        for idx, line in enumerate(lines):

            data = line.split();

            # evaluate with half precision where possible
            with amp.autocast():
                if feat_dir == '':
                    ref_feat = feats[data[1]].to(self.device)
                    com_feat = feats[data[2]].to(self.device)
                else:
                    ref_feat = torch.load(os.path.join(feat_dir,filedict[data[1]])).to(self.device)
                    com_feat = torch.load(os.path.join(feat_dir,filedict[data[2]])).to(self.device)

                if self.__test_normalize__:
                    ref_feat = F.normalize(ref_feat, p=2, dim=1)
                    com_feat = F.normalize(com_feat, p=2, dim=1)

                dist = F.pairwise_distance(ref_feat.unsqueeze(-1).expand(-1,-1,num_eval), com_feat.unsqueeze(-1).expand(-1,-1,num_eval).transpose(0,2)).detach().cpu().numpy();

                score = -1 * np.mean(dist);

                all_scores.append(score);  
                all_labels.append(int(data[0]));

            if idx % print_interval == 0:
                telapsed = time.time() - tstart
                print(f"Computing {idx}/{total_length}: {(idx/telapsed):.2f} Hz")

        if feat_dir != '':
            print(' Deleting temporary files.')
            shutil.rmtree(feat_dir)

        return (all_scores, all_labels);


    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Update learning rate
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def updateLearningRate(self, alpha):

        learning_rate = []
        for param_group in self.__optimizer__.param_groups:
            param_group['lr'] = param_group['lr']*alpha
            learning_rate.append(param_group['lr'])

        return learning_rate;


    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Save parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def saveParameters(self, path):
        
        torch.save(self.state_dict(), path);
        
        
    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Save model
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def saveModel(self, path):
        
        torch.save(self, path);



    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Load parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def loadParameters(self, path):

        self_state = self.state_dict();
        loaded_state = torch.load(path);
        for name, param in loaded_state.items():
            origname = name;
            if name not in self_state:
                name = name.replace("module.", "");

                if name not in self_state:
                    print("%s is not in the model."%origname);
                    continue;

            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()));
                continue;

            self_state[name].copy_(param);
