"""
    ...
"""
# Imports
import os
from datetime import datetime
import argparse
import numpy as np
import torch
from trainer import Trainer
from utils import _logger
from model import TFC, TargetClassifier
from dataloader import data_generator
from config_files.ECG_Configs import Configs

# Args selections
start_time = datetime.now()
parser = argparse.ArgumentParser()

# Model parameters
home_dir = os.getcwd()


# Set up command line arguments and create parser
parser.add_argument('--run_description', default='run1', type=str,
                    help='Experiment Description')
parser.add_argument('--seed', default=42, type=int,
                    help='seed value')
parser.add_argument('--training_mode', default='fine_tune_test', type=str,
                    help='pre_train, fine_tune_test')
parser.add_argument('--pretrain_dataset', default='ECG', type=str,
                    help='Dataset of choice: ECG')
parser.add_argument('--target_dataset', default='MIT-BIH', type=str,
                    help='Dataset of choice: EMG, MIT-BIH, PTB-XL-Superclass, PTB-XL-Form, PTB-XL-Rhythm')
parser.add_argument('--logs_save_dir', default='../experiments_logs', type=str,
                    help='saving directory')
parser.add_argument('--device', default='cpu', type=str,
                    help='cpu or cuda')
parser.add_argument('--home_path', default=home_dir, type=str,
                    help='Project home directory')

args, unknown = parser.parse_known_args()

# Set up device
with_gpu = torch.cuda.is_available()
if with_gpu:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"We are using {device} now.")

# Set up paths, experiment description and loggers
pretrain_dataset = args.pretrain_dataset
target_data = args.target_dataset
experiment_description = str(pretrain_dataset) + '_2_' + str(target_data)

method = 'TF-C'
training_mode = args.training_mode
run_description = args.run_description
logs_save_dir = args.logs_save_dir
os.makedirs(logs_save_dir, exist_ok=True)

# Use ECG_Configs
# exec(f'from config_files.{pretrain_dataset}_Configs import Config as Configs')
configs = Configs()

# fix random seeds for reproducibility
SEED = args.seed
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

# Set up experiment log directory and initialize logger
experiment_log_dir = os.path.join(logs_save_dir, experiment_description, run_description,
                                  training_mode + f"_seed_{SEED}_2layertransformer")
# 'experiments_logs/Exp1/run1/train_linear_seed_0'
os.makedirs(experiment_log_dir, exist_ok=True)

# loop through domains
counter = 0
src_counter = 0

# Logging
log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
# 'experiments_logs/Exp1/run1/train_linear_seed_0/logs_14_04_2022_15_13_12.log'
logger = _logger(log_file_name)
logger.debug("=" * 45)
logger.debug('Pre-training Dataset: %s', pretrain_dataset)
logger.debug('Target (fine-tuning) Dataset: %s', target_data)
logger.debug('Method:  %s', method)
logger.debug('Mode: %s', training_mode)
logger.debug("=" * 45)

# Load datasets
sourcedata_path = f"../../datasets/{pretrain_dataset}"
target_data_path = f"../../datasets/{target_data}"
subset = True  # if subset=True, use a subset for debugging.
train_dl, valid_dl, test_dl = data_generator(sourcedata_path, target_data_path,
                                             configs, training_mode, subset=subset)
logger.debug("Data loaded ...")

# Load Model
# Here are two models, one basemodel, another is temporal contrastive model
TFC_model = TFC(configs).to(device)
classifier = TargetClassifier(configs).to(device)
temporal_contr_model = None

if training_mode == "fine_tune_test":
    # load saved model of this experiment
    load_from = os.path.join(os.path.join(logs_save_dir, experiment_description, run_description,
                             f"pre_train_seed_{SEED}_2layertransformer", "saved_models"))
    print("The loading file path", load_from)
    chkpoint = torch.load(os.path.join(load_from, "ckp_last.pt"), map_location=device)
    pretrained_dict = chkpoint["model_state_dict"]
    TFC_model.load_state_dict(pretrained_dict)

model_optimizer = torch.optim.Adam(TFC_model.parameters(), lr=configs.lr,
                                   betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=configs.lr,
                                        betas=(configs.beta1, configs.beta2), weight_decay=3e-4)

# Trainer
Trainer(TFC_model, model_optimizer, classifier,
        classifier_optimizer, train_dl, valid_dl,
        test_dl, device, logger,
        configs, experiment_log_dir, training_mode)

logger.debug("Training time is : %s", datetime.now() - start_time)
