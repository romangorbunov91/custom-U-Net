import os
import numpy as np
import torch
import json
import random
from pathlib import Path
from datetime import datetime
import argparse

from train import ResNetTrainer, UNetTrainer#, UNetBacboneTrainer
from utils.configer import Configer

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  # To have ~deterministic results
        torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--disable-cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--hypes', default=None, type=str,
                        dest='hypes', help='The file of the hyper parameters.')
    parser.add_argument('--phase', default='train', type=str,
                        dest='phase', help='The phase of module.')
    parser.add_argument('--gpu', default=[0, ], nargs='+', type=int,
                        dest='gpu', help='The gpu used.')
    parser.add_argument('--resume', default=None, type=str,
                        dest='resume', help='The path of pretrained model.')
    
    args = parser.parse_args()

    torch.autograd.set_detect_anomaly(True)
    configer = Configer(args)
    
    # Read config-files.
    hyperparameters_path = Path("./src/hyperparameters/")
    
    with open(hyperparameters_path / "config.json", "r") as f:
        configer.general_config = json.load(f)
    
    with open(hyperparameters_path / (str(configer.get('model_name')) + "-config.json"), "r") as f:
        configer.model_config = json.load(f)
    
    with open(hyperparameters_path / (str(configer.get('dataset_name')) + "-config.json"), "r") as f:
        configer.dataset_config = json.load(f)
    
    if configer.get('backbone_model_name') is not None:
        with open(hyperparameters_path / (str(configer.get('backbone_model_name')) + "-config.json"), "r") as f:
            configer.backbone_model_config = json.load(f)
        with open(hyperparameters_path / (str(configer.backbone_model_config.get('dataset_name')) + "-config.json"), "r") as f:
            configer.backbone_dataset_config = json.load(f)
        
    set_seed(configer.general_config.get("seed"))

    configer.device = configer.general_config.get("device").lower() if torch.cuda.is_available() else 'cpu'
    
    if configer.model_config.get('model_name') == "customResNet":
        configer.output_file_name = (
            f"{str(configer.model_config.get('model_name'))}_"
            f"{str(configer.model_config.get('layers_num'))}x"
            f"{str(configer.model_config.get('block_size'))}_"
            f"classes_{str(len(configer.dataset_config.get('selected_classes')))}"
        )
    elif configer.model_config.get('model_name') == "customUNet":
        if configer.model_config.get('backbone_model_name') is None:
            configer.output_file_name = (
                f"{str(configer.model_config.get('model_name'))}"
            )
        else:
            configer.output_file_name = (
                f"{str(configer.model_config.get('model_name'))}_"
                f"backbone_{str(configer.model_config.get('backbone_model_name'))}_"
                f"finetune_last_{str(configer.model_config.get('backbone_tune_epoch'))}_epochs"
            )
    else:
        raise NotImplementedError(f"Model not supported: {configer.model_config.get('model_name')}")

    if configer.model_config.get('model_name') == "customResNet":
        model = ResNetTrainer(configer)
        model.init_model()
        train_history, train_size, val_size, model_param_count = model.train()

        train_log = [
        {
            "epoch": train_history["epoch"][i],
            "train_loss": train_history["train_loss"][i],
            "train_accuracy": train_history["train_accuracy"][i],
            "val_loss": train_history["val_loss"][i],
            "val_accuracy": train_history["val_accuracy"][i],
            "lr": train_history["lr"][i],
        }
        for i in range(len(train_history["epoch"]))
        ]
        
        output_dict = {
            "metadata": {
                "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "model": {
                    "name": configer.model_config.get("model_name"),
                    "layers_num": configer.model_config.get("layers_num"),
                    "block_size": configer.model_config.get("block_size"),
                    "param_count": model_param_count,
                    "checkpoints_metric": configer.model_config.get("checkpoints_metric")
                    },
                "dataset": {
                    "name": configer.dataset_config.get("dataset_name"),
                    "img_size": configer.dataset_config.get("img_size"),
                    "train_size": train_size,
                    "val_size": val_size,
                    "class_size": len(configer.dataset_config.get("selected_classes")),
                    "selected_classes": configer.dataset_config.get("selected_classes")
                    },
                "device": configer.device,
                "seed": configer.general_config.get("seed"),
                "workers": configer.model_config.get("workers"),
                "batch_size": configer.model_config.get("batch_size"),
                "solver_type": configer.model_config.get("solver_type")
                },
            "summary": {
                "best_val_" + configer.model_config.get("checkpoints_metric"): max(train_history["val_" + configer.model_config.get("checkpoints_metric")]),
                "best_epoch": train_history["epoch"][train_history["val_" + configer.model_config.get("checkpoints_metric")].index(max(train_history["val_" + configer.model_config.get("checkpoints_metric")]))],
                "final_train_" + configer.model_config.get("checkpoints_metric"): train_history["train_" + configer.model_config.get("checkpoints_metric")][-1],
                "final_val_" + configer.model_config.get("checkpoints_metric"): train_history["val_" + configer.model_config.get("checkpoints_metric")][-1],
                },
            "train_log": train_log
        }

    elif configer.model_config.get('model_name') == "customUNet":
        if configer.model_config.get('backbone_model_name') is None:
            model = UNetTrainer(configer)
            model.init_model()
            train_history, train_size, val_size, model_param_count = model.train()

            train_log = [
            {
                "epoch": train_history["epoch"][i],
                "train_loss": train_history["train_loss"][i],
                "train_dice": train_history["train_dice"][i],
                "train_iou": train_history["train_iou"][i],
                "train_accuracy": train_history["train_accuracy"][i],
                "val_loss": train_history["val_loss"][i],
                "val_dice": train_history["val_dice"][i],
                "val_iou": train_history["val_iou"][i],
                "val_accuracy": train_history["val_accuracy"][i],
                "lr": train_history["lr"][i],
            }
            for i in range(len(train_history["epoch"]))
            ]
            backbone_model_param_count = None
            backbone_train_size = 0
            backbone_val_size = 0
            output_dict = {
                "metadata": {
                    "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
                    "model": {
                        "name": configer.model_config.get("model_name"),
                        "layers_num": configer.model_config.get("feature_list"),
                        "param_count": model_param_count
                        },
                    "backbone_model": {
                        "name": configer.model_config.get('backbone_model_name'),
                        "backbone_param_count": backbone_model_param_count
                        },
                    "dataset": {
                        "name": configer.dataset_config.get("dataset_name"),
                        "img_size": configer.dataset_config.get("img_size"),
                        "train_size": train_size,
                        "val_size": val_size
                        },
                    '''
                    "backbone_dataset": {
                        "name": configer.backbone_dataset_config.get("dataset_name"),
                        "img_size": configer.backbone_dataset_config.get("img_size"),
                        "train_size": backbone_train_size,
                        "val_size": backbone_val_size,
                        "class_size": len(configer.backbone_dataset_config.get("selected_classes")),
                        "selected_classes": configer.backbone_dataset_config.get("selected_classes")
                        },
                    '''
                    "device": configer.device,
                    "seed": configer.general_config.get("seed"),
                    "workers": configer.model_config.get("workers"),
                    "batch_size": configer.model_config.get("batch_size"),
                    "solver_type": configer.model_config.get("solver_type")
                    },
                "summary": {
                    "best_val_" + configer.model_config.get("checkpoints_metric"): max(train_history["val_" + configer.model_config.get("checkpoints_metric")]),
                    "best_epoch": train_history["epoch"][train_history["val_" + configer.model_config.get("checkpoints_metric")].index(max(train_history["val_" + configer.model_config.get("checkpoints_metric")]))],
                    "final_train_" + configer.model_config.get("checkpoints_metric"): train_history["train_" + configer.model_config.get("checkpoints_metric")][-1],
                    "final_val_" + configer.model_config.get("checkpoints_metric"): train_history["val_" + configer.model_config.get("checkpoints_metric")][-1],
                    },
                "train_log": train_log
            }
        '''
        else:
            model = UNetBacboneTrainer(configer)
            model.init_model()
            train_history, train_size, val_size, model_param_count = model.train()

            train_log = [
            {
                "epoch": train_history["epoch"][i],
                "train_loss": train_history["train_loss"][i],
                "train_dice": train_history["train_dice"][i],
                "train_iou": train_history["train_iou"][i],
                "train_accuracy": train_history["train_accuracy"][i],
                "val_loss": train_history["val_loss"][i],
                "val_dice": train_history["val_dice"][i],
                "val_iou": train_history["val_iou"][i],
                "val_accuracy": train_history["val_accuracy"][i],
                "lr": train_history["lr"][i],
            }
            for i in range(len(train_history["epoch"]))
            ]
            
            output_dict = {
                "metadata": {
                    "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
                    "model": {
                        "name": configer.get("model", "name"),
                        "param_count": model_param_count
                        },
                    "backbone_model": {
                        "name": configer.get("model", "name"),
                        "param_count": model_param_count
                        },
                    "dataset": {
                        "name": configer.get("dataset", "name"),
                        "img_size": configer.get("dataset", "img_size"),
                        "train_size": train_size,
                        "val_size": val_size
                        },
                    "backbone_dataset": {
                        "name": configer.get("dataset", "name"),
                        "img_size": configer.get("dataset", "img_size"),
                        "train_size": train_size,
                        "val_size": val_size,
                        "class_size": len(configer.get("dataset", "selected_classes")),
                        "selected_classes": configer.get("dataset", "selected_classes")
                        },
                    "device": configer.device,
                    "workers": configer.get("data", "workers"),
                    "batch_size": configer.get("data", "batch_size"),
                    "solver": configer.get("solver", "type"),
                    "seed": configer.get("seed")
                    },
                "summary": {
                    "best_val_dice": max(train_history["val_dice"]),
                    "best_epoch": train_history["epoch"][train_history["val_dice"].index(max(train_history["val_dice"]))],
                    "final_train_dice": train_history["train_dice"][-1],
                    "final_val_dice": train_history["val_dice"][-1],
                    },
                "train_log": train_log
            }
        '''
    else:
        raise NotImplementedError(f"Model not supported: {configer.model_config.get('model_name')}")
    
    
    logs_dir = Path(configer.general_config.get("logs_dir"))
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    with open(logs_dir / (configer.output_file_name + '.json'), "w") as f:
        json.dump(output_dict, f, indent=4)