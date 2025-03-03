import os
import sys
from os.path import dirname
DOSSIER_COURRANT = dirname(os.path.abspath(__file__))
ROOT_FOLDER = dirname(dirname(dirname(DOSSIER_COURRANT)))
sys.path.append(ROOT_FOLDER)
from .trainer_line_ctc import TrainerLineCTC
from .models_line_ctc import Decoder
from models.encoder import FCN_Encoder
from torch.optim import Adam
from dataset.transforms import line_aug_config
from basic.scheduler import exponential_dropout_scheduler, exponential_scheduler
from OCR.ocr_dataset_manager import OCRDataset, OCRDatasetManager
import torch.multiprocessing as mp
import torch
import numpy as np
import random


def train_and_test(rank, params):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    params["training_params"]["ddp_rank"] = rank
    model = TrainerLineCTC(params)
    model.load_model()

    # Model trains until max_time_training or max_nb_epochs is reached
    model.train()

    # load weights giving best CER on valid set
    model.params["training_params"]["load_epoch"] = "best"
    model.load_model()

    # compute metrics on train, valid and test sets (in eval conditions)
    metrics = ["cer", "wer", "time", ]
    for dataset_name in params["dataset_params"]["datasets"].keys():
        for set_name in ["test", "valid", "train", ]:
            model.predict("{}-{}".format(dataset_name, set_name), [(dataset_name, set_name), ], metrics, output=True)


def main():
    dataset_name = "READ_2016"  # ["RIMES", "READ_2016"]
    dataset_level = "syn_line"
    params = {
        "dataset_params": {
            "dataset_manager": OCRDatasetManager,
            "dataset_class": OCRDataset,
            "datasets": {
                dataset_name: "../../../Datasets/formatted/{}_{}".format(dataset_name, dataset_level),
            },
            "train": {
                "name": "{}-train".format(dataset_name),
                "datasets": [(dataset_name, "train"), ],
            },
            "valid": {
                "{}-valid".format(dataset_name): [(dataset_name, "valid"), ],
            },
            "config": {
                "load_in_memory": True,  # Load all images in CPU memory
                "worker_per_gpu": 8,  # Num of parallel processes per gpu for data loading
                "width_divisor": 8,  # Image width will be divided by 8
                "height_divisor": 32,  # Image height will be divided by 32
                "padding_value": 0,  # Image padding value
                "padding_token": 1000,  # Label padding value (None: default value is chosen)
                "padding_mode": "br",  # Padding at bottom and right
                "charset_mode": "CTC",  # add blank token
                "constraints": ["CTC_line", ],  # Padding for CTC requirements if necessary
                "normalize": True,  # Normalize with mean and variance of training dataset
                "padding": {
                    "min_height": "max",  # Pad to reach max height of training samples
                    "min_width": "max",  # Pad to reach max width of training samples
                    "min_pad": None,
                    "max_pad": None,
                    "mode": "br",  # Padding at bottom and right
                    "train_only": False,  # Add padding at training time and evaluation time
                },
                "preprocessings": [
                    {
                        "type": "to_RGB",
                        # if grayscale image, produce RGB one (3 channels with same value) otherwise do nothing
                    },
                ],
                # Augmentation techniques to use at training time
                "augmentation": line_aug_config(0.9, 0.1),
                #
                "synthetic_data": {
                    "mode": "line_hw_to_printed",
                    "init_proba": 1,
                    "end_proba": 1,
                    "num_steps_proba": 1e5,
                    "proba_scheduler_function": exponential_scheduler,
                    "config": {
                        "background_color_default": (255, 255, 255),
                        "background_color_eps": 15,
                        "text_color_default": (0, 0, 0),
                        "text_color_eps": 15,
                        "font_size_min": 30,
                        "font_size_max": 50,
                        "color_mode": "RGB",
                        "padding_left_ratio_min": 0.02,
                        "padding_left_ratio_max": 0.1,
                        "padding_right_ratio_min": 0.02,
                        "padding_right_ratio_max": 0.1,
                        "padding_top_ratio_min": 0.02,
                        "padding_top_ratio_max": 0.2,
                        "padding_bottom_ratio_min": 0.02,
                        "padding_bottom_ratio_max": 0.2,
                    },
                },
            }
        },

        "model_params": {
            # Model classes to use for each module
            "models": {
                "encoder": FCN_Encoder,
                "decoder": Decoder,
            },
            "transfer_learning": None,
            "input_channels": 3,  # 1 for grayscale images, 3 for RGB ones (or grayscale as RGB)
            "enc_size": 256,
            "dropout_scheduler": {
                "function": exponential_dropout_scheduler,
                "T": 5e4,
            },
            "dropout": 0.5,
        },

        "training_params": {
            "output_folder": "FCN_read_2016_line_syn",  # folder names for logs and weigths
            "max_nb_epochs": 10000,  # max number of epochs for the training
            "max_training_time": 3600 * 24 * 1.9,  # max training time limit (in seconds)
            "load_epoch": "last",  # ["best", "last"], to load weights from best epoch or last trained epoch
            "interval_save_weights": None,  # None: keep best and last only
            "use_ddp": False,  # Use DistributedDataParallel
            "use_amp": True,  # Enable automatic mix-precision
            "nb_gpu": torch.cuda.device_count(),
            "batch_size": 16,  # mini-batch size per GPU
            "optimizers": {
                "all": {
                    "class": Adam,
                    "args": {
                        "lr": 0.0001,
                        "amsgrad": False,
                    }
                }
            },
            "lr_schedulers": None,  # Learning rate schedulers
            "eval_on_valid": True,  # Whether to eval and logs metrics on validation set during training or not
            "eval_on_valid_interval": 2,  # Interval (in epochs) to evaluate during training
            "focus_metric": "cer",  # Metrics to focus on to determine best epoch
            "expected_metric_value": "low",  # ["high", "low"] What is best for the focus metric value
            "set_name_focus_metric": "{}-valid".format(dataset_name),  # Which dataset to focus on to select best weights
            "train_metrics": ["loss_ctc", "cer", "wer"],  # Metrics name for training
            "eval_metrics": ["loss_ctc", "cer", "wer"],  # Metrics name for evaluation on validation set during training
            "force_cpu": False,  # True for debug purposes to run on cpu only
        },
    }

    if params["training_params"]["use_ddp"] and not params["training_params"]["force_cpu"]:
        mp.spawn(train_and_test, args=(params,), nprocs=params["training_params"]["nb_gpu"])
    else:
        train_and_test(0, params)


if __name__ == "__main__":
    main()