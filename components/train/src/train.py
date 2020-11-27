#!/usr/bin/python
#-*- coding: utf-8 -*-

import sys, os

# Add common src dir to python import path (varies between runs on and
# off the training cluster)
sys.path.insert(0, os.getenv('VOX_COMMON_SRC_DIR'))

import pwd
import pdb
import glob
import time
import yaml
import wandb
import numpy
import torch
import random
import google
import subprocess
from pathlib import Path
import time, os, argparse, socket
from SpeakerNet import SpeakerNet
from IterableTrainDataset import IterableTrainDataset
from baseline_misc.tuneThreshold import tuneThresholdfromScore
from utils.data_utils import download_gcs_dataset, extract_gcs_dataset, \
                     transcode_gcs_dataset, get_loc_paths_from_gcs_dataset,\
                     download_blob, upload_blob

# @brief Generate a unique run ID when not run in kubeflow (kubeflow passes
#        its own default run ID) in order to store training artifacts
#        for resume after preemption
# @note To manually resume a run outside of kubeflow, pass the run ID
#       printed in the run with the "--run-id" flag
def gen_run_id():
    user_id = pwd.getpwuid( os.getuid() )[ 0 ]
    wandb_id = wandb.util.generate_id()
    return f"{user_id}-{wandb_id}"

parser = argparse.ArgumentParser(description = "SpeakerNet");

## New args to support running on kubernetes/kubeflow
# @note "tmp" denotes that this output data will not be captured by
#       the kubeflow pipeline or made available to downstream components
# set --data-bucket in order to fetch lists and data from GCS before reading
# them from local filesystem

# temporary/internal outputs
parser.add_argument('--data-bucket', type=str)
parser.add_argument('--save-tmp-data-to', type=str, default="./tmp/data/")
parser.add_argument('--skip-data-fetch', action='store_true')
parser.add_argument('--reset-training', action='store_true', help='Reset \
        training to first epoch, regardless of previously saved model checkpoints')
parser.add_argument('--save-tmp-model-to', type=str, default="./tmp/model/");
parser.add_argument('--save-tmp-results-to', type=str, default="./tmp/results/");
parser.add_argument('--save-tmp-feats-to', type=str, default="./tmp/feats/");
parser.add_argument('--save-tmp-wandb-to', type=str, default="./tmp/");

parser.add_argument('--set-seed', action='store_true')
parser.add_argument('--no-cuda', action='store_true', help="Flag to disable cuda for this run")

parser.add_argument('--checkpoint-bucket', type=str,
        default="voxsrc-2020-checkpoints-dev");
parser.add_argument('--run-id', type=str, default=f"{gen_run_id()}");

# new training params
parser.add_argument('--gaussian-noise-std', type=float, default=.9,
        help="Standard deviation of gaussian noise used to augment utterance "
             "spectrogram training data");

# permanent/component outputs
parser.add_argument('--save-model-to', type=str, default="./out/model.txt")

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

args = parser.parse_args();

# set random seeds
# @TODO any reason to use BOTH 'random' and 'numpy.random'?
if args.set_seed:
    print("train: Using fixed random seed")
    random.seed(0)
    numpy.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

# touch wandb tmp dir
print(f"train: Creating directory {args.save_tmp_wandb_to}")
Path(args.save_tmp_wandb_to).mkdir(parents=True, exist_ok=True)

wandb.init(project="initial-setup", entity="voxceleb-fairness", config=args,
        id=args.run_id, resume="allow", dir=args.save_tmp_wandb_to)

train_list, test_list, train_path, test_path = [None, None,
        None, None]

# make directories for tmp data
if not(os.path.exists(args.save_tmp_data_to)):
    os.makedirs(args.save_tmp_data_to)
if not(os.path.exists(args.save_tmp_model_to)):
    os.makedirs(args.save_tmp_model_to)

# Install data from GCS, skipping download if already downladed, and skipping
# extraction if already downloaded and extracted
print("train: Installing dataset from GCS")
# @TODO mimic the --install-local-dataset function in
#       data/utils.py, using the newer functions that it invokes
#       in common/src/utils/data_utils.py

# download, extract, transcode (compressed AAC->WAV) dataset
blobs = [args.train_list, args.test_list, args.train_path,
        args.test_path]
download_gcs_dataset(args.data_bucket, args.save_tmp_data_to, blobs)
# @TODO make extract_gcs_dataset take blob names as well
extract_gcs_dataset(args, use_pigz=True)
# set new lists and data paths
train_list, test_list, train_path, test_path \
    = get_loc_paths_from_gcs_dataset(args.save_tmp_data_to, blobs)

# init directories
# temporary / internal output directories
tmp_output_dirs = [args.save_tmp_model_to, args.save_tmp_results_to,
        args.save_tmp_feats_to]
# directories of parmanent / component output artifacts
# @TODO this working?
output_dirs = [os.path.dirname(args.save_model_to)]

for d in (tmp_output_dirs + output_dirs):
    if not(os.path.exists(d)):
        os.makedirs(d)

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

it          = 1;
prevloss    = float("inf");
sumloss     = 0;

# Check for training meta data from a previously preempted run
METADATA_NAME = 'metadata.yaml'
metadata_gcs_src_path = os.path.join(args.run_id, METADATA_NAME)
metadata_file_dst_path = os.path.join(args.save_tmp_model_to, METADATA_NAME)
default_metadata = {'is_done': False, 'val_EER': 0}
metadata = default_metadata

def save_model_kf_output_artifact(model, epoch, dst):
    final_model_name = "model%09d.model"%epoch
    Path(dst).mkdir(parents=True, exist_ok=True)
    final_model_filename = os.path.join(dst, final_model_name)
    print(f"train: Saving KF output artifact {final_model_name} to {dst}")
    model.saveParameters(final_model_filename);

# fetch metadata and latest model, if available
try:
    download_blob(args.checkpoint_bucket, metadata_gcs_src_path,
            metadata_file_dst_path)
    print("train: Downloaded previous training metadata")
    with open(metadata_file_dst_path, 'r') as f:
        try:
            metadata = yaml.safe_load(f)
            print(f"train: Loaded previous training metadata: {metadata}")
            # grab the latest model name (corresponding to the last epoch)
            latest_model_name = metadata['latest_model_name']
            # download the model
            model_gcs_src_path = os.path.join(args.run_id, latest_model_name)
            model_file_dst_path = os.path.join(args.save_tmp_model_to, latest_model_name)
            try:
                download_blob(args.checkpoint_bucket, model_gcs_src_path,
                        model_file_dst_path)
                print("train: Downloaded a saved model")
                # load the saved model's params into the model class
                s.loadParameters(model_file_dst_path);
                print(f"train: Loaded model params from path {model_file_dst_path}");
                it = int(os.path.splitext(os.path.basename(model_file_dst_path))[0][5:]) + 1
            except google.cloud.exceptions.NotFound:
                print("train: No saved model found")
        except yaml.YAMLError as exc:
            print(exc)
            metadata = default_metadata
except google.cloud.exceptions.NotFound:
    print("train: No previous training metadata found**")

# exit if previous training run finished
if 'is_done' in metadata and metadata['is_done']:
    save_model_kf_output_artifact(s, metadata['num_epochs'],
                                  args.save_model_to)
    print("train: Terminating... training for this run has already completed")
    quit()

if(args.initial_model != ""):
    raise "Error: TODO"
    s.loadParameters(args.initial_model);
    print(f"train: Loaded model params from path {args.initial_model}");

for ii in range(0,it-1):
    if ii % args.lr_decay_interval == 0:
        clr = s.updateLearningRate(args.lr_decay) 

## Evaluation code
if args.eval == True:
    scores, labels = s.evaluate_on(test_list, test_path)
    result = tuneThresholdfromScore(scores, labels, [1, 0.1]);
    print(f"train: Initial EER is {result[1]:2.4f}")

    quit();

for items in vars(args):
    print(items, vars(args)[items]);

## Assertion
gsize_dict  = {'proto':args.nSpeakers, 'triplet':2, 'contrastive':2,
        'softmax':1, 'amsoftmax':1, 'aamsoftmax':1, 'ge2e':args.nSpeakers,
        'angleproto':args.nSpeakers}

assert args.trainfunc in gsize_dict
assert gsize_dict[args.trainfunc] <= 100

# new data loading
voxceleb_2_iterable_dataset = IterableTrainDataset(train_list,
        gSize=gsize_dict[args.trainfunc], new_train_path=train_path,
        **vars(args))

# @note batch_size = None bypasses the default batching mechanism
train_loader = torch.utils.data.DataLoader(voxceleb_2_iterable_dataset,
        batch_size = None, num_workers=args.n_data_loader_thread)

clr = s.updateLearningRate(1)

# touch the output file/dir
print(f"Creating parent dir for path={args.save_tmp_model_to}")
Path(args.save_tmp_model_to).parent.mkdir(parents=True, exist_ok=True)

wandb_log = {'epoch': 0, 'loss': 0, 'train_EER': 0, 'clr': max(clr),
        'val_EER': metadata['val_EER']}

def iter_info(epoch):
    return f"train: {time.strftime('%Y-%m-%d %H:%M:%S')} | Epoch {epoch}"

# @TODO rename 'it'
for epoch in range(it, args.max_epoch):
    learning_rate = max(clr)
    print(f"{iter_info(epoch)}: Training {args.model} with LR {learning_rate}")

    # train
    loss, train_eer = s.train_on(loader=train_loader,
            data_length=len(voxceleb_2_iterable_dataset));

    epoch_results = f"LR {learning_rate}, Train T1/EER {float(train_eer)}, Train loss {loss}"

    # validate model
    if epoch % args.test_interval == 0:
        print(f"{iter_info(epoch)}: Evaluating model on test set")

        # compute EER on test set
        scores, labels = s.evaluate_on(test_list, test_path)
        result = tuneThresholdfromScore(scores, labels, [1, 0.1]);
        val_eer = float(round(result[1],5))

        epoch_results += f", Val EER {val_eer}"
        wandb_log.update({'val_EER': val_eer})

    print(f"{iter_info(epoch)}: {epoch_results}")

    wandb_log.update({'epoch': epoch, 'loss': loss, 'train_EER': train_eer, 'clr': learning_rate})
    wandb.log(wandb_log)

    # save model file for this epoch
    model_name = "model%09d.model"%epoch
    model_filename = os.path.join(args.save_tmp_model_to, model_name)
    s.saveParameters(model_filename);
    # update metadata
    metadata['latest_model_name'] = model_name
    metadata['num_epochs'] = epoch
    metadata['val_EER'] = wandb_log['val_EER']
    # dump metadata to yaml file
    with open(metadata_file_dst_path, 'w') as f:
        try:
            yaml.dump(metadata, f)
            print("Saved current training metadata")
        except yaml.YAMLError as exc:
            print(exc)
    # upload model to GCS
    model_gcs_dst_path = os.path.join(args.run_id, model_name)
    model_file_src_path = os.path.join(args.save_tmp_model_to, model_name)
    upload_blob(args.checkpoint_bucket, model_gcs_dst_path,
            model_file_src_path)
    # upload metadata to GCS
    metadata_gcs_dst_path = metadata_gcs_src_path
    metadata_file_src_path = metadata_file_dst_path
    upload_blob(args.checkpoint_bucket, metadata_gcs_dst_path,
            metadata_file_src_path)

    # update learning rate
    if epoch % args.lr_decay_interval == 0:
        clr = s.updateLearningRate(args.lr_decay)

# save "done" to metadata so it restarts on retry
metadata['is_done'] = True

# dump metadata to yaml file
with open(metadata_file_dst_path, 'w') as f:
    try:
        yaml.dump(metadata, f)
        print("Saved current training metadata")
    except yaml.YAMLError as exc:
        print(exc)

# upload metadata to GCS
metadata_gcs_dst_path = metadata_gcs_src_path
metadata_file_src_path = metadata_file_dst_path
upload_blob(args.checkpoint_bucket, metadata_gcs_dst_path,
        metadata_file_src_path)

# write the final model as a kubeflow output artifact
save_model_kf_output_artifact(s, metadata['num_epochs'],
                              args.save_model_to)
