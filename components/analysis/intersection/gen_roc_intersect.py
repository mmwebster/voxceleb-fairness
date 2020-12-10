import sys, os
sys.path.insert(0, '/home/jupyter/voxceleb-fairness/common/')
sys.path.insert(0, '/home/jupyter/voxceleb-fairness/components/train/src/')

import pwd
import pdb
import glob
import time
import yaml
import wandb
import numpy
import torch
import json
import random
import google
from sklearn import metrics
import matplotlib.pyplot as plt
import subprocess
from pathlib import Path
import time, os, argparse, socket
from SpeakerNet import SpeakerNet
from IterableTrainDataset import IterableTrainDataset
from baseline_misc.tuneThreshold import tuneThresholdfromScore
from utils.data_utils import download_gcs_dataset, extract_gcs_dataset, \
                     transcode_gcs_dataset, get_loc_paths_from_gcs_dataset,\
                     download_blob, upload_blob

parser = argparse.ArgumentParser(description = "SpeakerNet");

parser.add_argument('--set-seed', action='store_true')
parser.add_argument('--no-cuda', action='store_true', help="Flag to disable cuda for this run")

parser.add_argument('--checkpoint-bucket', type=str,
        default="voxsrc-2020-checkpoints-dev");

# new training params
parser.add_argument('--gaussian-noise-std', type=float, default=.9,
        help="Standard deviation of gaussian noise used to augment utterance "
             "spectrogram training data");

## Data loader
parser.add_argument('--max_frames', type=int, default=200,  help='Input length to the network');
parser.add_argument('--batch_size', type=int, default=200,  help='Batch size');
# @TODO figure out max eval batch size for this on V100
parser.add_argument('--eval_batch_size', type=int, default=85,  help='Batch size for loading validation data for model eval');
# ^^^ use --batch_size=30 for small datasets that can't fill an entire 200 speaker pair/triplet batch
parser.add_argument('--max_seg_per_spk', type=int, default=100, help='Maximum number of utterances per speaker per epoch');
parser.add_argument('--n-data-loader-thread', type=int, default=7, help='Number of loader threads');

## Training details
# @TODO disentangle learning rate decay from validation
parser.add_argument('--test_interval', type=int, default=10, help='Test and save every [test_interval] epochs');
parser.add_argument('--max_epoch',      type=int, default=100, help='Maximum number of epochs');
# ^^^ use --max_epoch=1 for local testing
parser.add_argument('--trainfunc', type=str, default="angleproto",    help='Loss function');
parser.add_argument('--optimizer', type=str, default="adam", help='sgd or adam');

## Learning rates
parser.add_argument('--lr', type=float, default=0.001,      help='Learning rate');
parser.add_argument('--lr_decay_interval', type=int, default=10, help='Reduce the learning rate every [lr_decay_interval] epochs');
parser.add_argument("--lr_decay", type=float, default=0.95, help='Learning rate decay every [test_interval] epochs');

## Loss functions
parser.add_argument("--hard_prob", type=float, default=0.5, help='Hard negative mining probability, otherwise random, only for some loss functions');
parser.add_argument("--hard_rank", type=int, default=10,    help='Hard negative mining rank in the batch, only for some loss functions');
parser.add_argument('--margin', type=float,  default=0.3,     help='Loss margin, only for some loss functions');
parser.add_argument('--scale', type=float,   default=30,    help='Loss scale, only for some loss functions');
parser.add_argument('--nSpeakers', type=int, default=5994,  help='Number of speakers in the softmax layer for softmax-based losses, utterances per speaker per iteration for other losses');

## Load and save
parser.add_argument('--initial_model',  type=str, default="", help='Initial model weights');
parser.add_argument('--save_path',      type=str, default="/tmp/data/exp1", help='Path for model and logs');

## Training and test data
parser.add_argument('--train_list', type=str, help='Train list');
parser.add_argument('--test_list',  type=str, help='Evaluation list');
parser.add_argument('--train_path', type=str, default="voxceleb2", help='Absolute path to the train set');
parser.add_argument('--test_path',  type=str, default="voxceleb1", help='Absolute path to the test set');

## For test only
parser.add_argument('--eval', dest='eval', action='store_true', help='Eval only')

## Model definition
parser.add_argument('--model', type=str,        default="ResNetSE34L",     help='Name of model definition');
parser.add_argument('--encoder_type', type=str, default="SAP",  help='Type of encoder');
parser.add_argument('--nOut', type=int,         default=512,    help='Embedding size in the last FC layer');

args = parser.parse_args("");


# set random seeds
# @TODO any reason to use BOTH 'random' and 'numpy.random'?
if args.set_seed:
    print("train: Using fixed random seed")
    random.seed(0)
    numpy.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)


# set torch device to cuda or cpu
cuda_avail = torch.cuda.is_available()
print(f"train: Cuda available: {cuda_avail}")
use_cuda = cuda_avail and not args.no_cuda
print(f"train: Using cuda: {use_cuda}")
device = torch.device("cuda" if use_cuda else "cpu")
print(f"train: Torch version: {torch.__version__}")
print(f"train: Cuda version: {torch.version.cuda}")

## Load models
s = SpeakerNet(device, **vars(args));

s.loadParameters("/home/jupyter/model000000029.model")

spectrogram_feats_path = "/home/jupyter/voxceleb-fairness/data/datasets/full/vox1_full_feats_milo_webster-19rvuxfu"
list_files_base_path = "/home/jupyter/voxceleb-fairness/data/lists/intersect/"

# full paths to test lists
balanced_test_list_paths = glob.glob(os.path.join(list_files_base_path, "*balanced.txt"))

# names of each test list
balanced_test_pair_names = [str(x).split('vox1_')[1].split('_balanced.txt')[0] for x in balanced_test_list_paths]

print(f"Generating ROC data for {len(balanced_test_list_paths)} test lists")
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(121)
ax.set_xlabel("FPR")
ax.set_ylabel("TPR")

fpr_tpr_and_thresholds = []

# generate scores and labels by evaluating the lists against the model
for i, test_list_path in enumerate(balanced_test_list_paths):
    # evaluate model
    scores, labels = s.evaluate_on(test_list_path, spectrogram_feats_path)
    print(f"Evaluated scores/labels for {balanced_test_pair_names[i]}")
    
    # compute metrics
    fpr_tpr_and_thresholds.append(metrics.roc_curve(labels, scores, pos_label=1))
    
    # plot for sanity
    ax.scatter(fpr_tpr_and_thresholds[-1][0], fpr_tpr_and_thresholds[-1][1],
               label=balanced_test_pair_names[i], marker='.')

ax.set_aspect(1)
fig.legend(loc='center right')
#plt.show()
plt.savefig('roc_intersect')

# pack metrics for each test list into a dictionary
results = {}
for i, data in enumerate(fpr_tpr_and_thresholds):
    name = balanced_test_pair_names[i]
    results[name + "_fpr"] = data[0].tolist()
    results[name + "_tpr"] = data[1].tolist()
    results[name + "_thresholds"] = data[2].tolist()
    
# save the data as json
dest_file_path = '/home/jupyter/voxceleb-fairness/data/roc/roc_intersect.json'
with open(dest_file_path, 'w') as fp:
    json.dump(results, fp)