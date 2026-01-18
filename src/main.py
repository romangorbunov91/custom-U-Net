import os
import numpy as np
import torch
import json
import random
from pathlib import Path
from datetime import datetime
import argparse

from train import ResNet18Trainer, UNetTrainer, UNetBacboneTrainer
from utils.configer import Configer

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
    
    seed = configer.get("seed")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True  # To have ~deterministic results

    configer.device = configer.get("device").lower() if torch.cuda.is_available() else 'cpu'
    
    if configer.get('model', 'name') == "customResNet18":
        configer.output_file_name = (
            f"{str(configer.get('model', 'name'))}_"
            f"{str(configer.get('model', 'layers_num'))}x"
            f"{str(configer.get('model', 'block_size'))}_"
            f"classes_{str(len(configer.get('dataset', 'selected_classes')))}"
        )
    elif configer.get('model', 'name') == "customUNet":
        if configer.get('model', 'backbone') is None:
            configer.output_file_name = (
                f"{str(configer.get('model', 'name'))}"
            )
        else:
            configer.output_file_name = (
                f"{str(configer.get('model', 'name'))}_"
                f"backbone_{str(configer.get('model', 'backbone'))}_"
                f"finetune_last_{str(configer.get('backbone_tune_epoch'))}_epochs"
            )
    else:
        raise NotImplementedError(f"Model not supported: {configer.get('model', 'name')}")

    if configer.get('model', 'name') == "customResNet18":
        model = ResNet18Trainer(configer)
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
                    "name": configer.get("model", "name"),
                    "layers_num": configer.get("model", "layers_num"),
                    "block_size": configer.get("model", "block_size"),
                    "param_count": model_param_count
                    },
                "dataset": {
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
                "best_val_acc": max(train_history["val_accuracy"]),
                "best_epoch": train_history["epoch"][train_history["val_accuracy"].index(max(train_history["val_accuracy"]))],
                "final_train_acc": train_history["train_accuracy"][-1],
                "final_val_acc": train_history["val_accuracy"][-1],
                },
            "train_log": train_log
        }

    elif configer.get('model', 'name') == "customUNet":
        if configer.get('model', 'backbone') is None:
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
    else:
        raise NotImplementedError(f"Model not supported: {configer.get('model', 'name')}")
    
    
    logs_dir = Path(configer.get("checkpoints", "logs_dir"))
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    with open(logs_dir / (configer.output_file_name + '.json'), "w") as f:
        json.dump(output_dict, f, indent=4)