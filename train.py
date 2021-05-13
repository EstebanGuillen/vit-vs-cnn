
from __future__ import print_function

import pytorch_lightning as pl
import mlflow
import mlflow.pytorch
import os
import torch
from argparse import ArgumentParser
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.metrics.functional import accuracy
from torch.nn import functional as F
#from torch.utils.data import DataLoader, random_split

#from torchvision import datasets, transforms, models


import glob
from itertools import chain
#import os
import random
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from linformer import Linformer
from PIL import Image
#from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms,models
from tqdm.notebook import tqdm
from skimage import io

#from pytorch_pretrained_vit import ViT
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import math

image_size = 384
root_dir = '/nfs/research/ejguill/data/x_ray_data/all_images/'

train_list = '/nfs/research/ejguill/data/x_ray_data/train.csv'
valid_list = '/nfs/research/ejguill/data/x_ray_data/val.csv'
test_list = '/nfs/research/ejguill/data/x_ray_data/test.csv'

run_dir = '/nfs/research/ejguill/data/x_ray_data/runs/'


class X_RayDataset(Dataset):
    
    def __init__(self, csv_file, root_dir, transform=None):
        
        self.gnd_truth_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.gnd_truth_frame)
    
    def __getitem__(self, idx):
        
        file_name = self.gnd_truth_frame.iloc[idx,0]
        label = self.gnd_truth_frame.iloc[idx,1]
        
        
        img = Image.open(self.root_dir + file_name)
        img = img.convert('RGB')
        img_transformed = self.transform(img)
        
        return img_transformed, label



class X_RayDataModule(pl.LightningDataModule):
    def __init__(self, **kwargs):
        super(X_RayDataModule, self).__init__()
        self.df_train = None
        self.df_val = None
        self.df_test = None
        self.train_data_loader = None
        self.val_data_loader = None
        self.test_data_loader = None
        self.args = kwargs

        

    def setup(self, stage=None):
        if self.args["data_aug"] == "yes":
            train_transforms = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(degrees=90),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])])
        else:
            train_transforms = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])])

        val_transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

        test_transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
        
        train_data = X_RayDataset(train_list, root_dir=root_dir, transform=train_transforms)
        valid_data = X_RayDataset(valid_list, root_dir=root_dir, transform=val_transforms)
        test_data = X_RayDataset(test_list, root_dir=root_dir, transform=test_transforms)

        self.df_train = train_data
        self.df_val = valid_data
        self.df_test = test_data
    
    def train_set_size(self):
        return len(self.df_train)

    def create_data_loader(self, df, shuffle=True):
        
        return DataLoader(
            df, batch_size=self.args["batch_size"], num_workers=self.args["num_workers"],shuffle=shuffle
        )

    def train_dataloader(self):
        
        return self.create_data_loader(self.df_train)

    def val_dataloader(self):
       
        return self.create_data_loader(self.df_val,shuffle=False)

    def test_dataloader(self):
        
        return self.create_data_loader(self.df_test,shuffle=False)


def create_vit_model(pre_trained=True):
    model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_384', pretrained=pre_trained)
    num_ftrs = model.head.in_features
    model.head = nn.Linear(num_ftrs, 2)
    return model

def create_cnn_model(pre_trained=True):
    model = models.resnet152(pretrained=pre_trained)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    return model

def create_dense_model(pre_trained=True):
    model = models.densenet201(pretrained=pre_trained)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    return model

class LightningX_RayClassifier(pl.LightningModule):
    def __init__(self, **kwargs):
        
        super(LightningX_RayClassifier, self).__init__()
        self.args = kwargs
        self.pre_trained = self.args["pre_trained"]
        if self.pre_trained == "yes":
            pre_train = True
        else:
            pre_train = False
        if self.args["arch"] == "vit":
            self.model = create_vit_model(pre_train)
        elif self.args["arch"] == "dense":
            self.model = create_dense_model(pre_train)
        else:
            self.model = create_cnn_model(pre_train)
        

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
       
        output = self.model(x)

        return output

    
    def training_step(self, train_batch, batch_idx):
        
        x, y = train_batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        
        _, y_hat = torch.max(logits, dim=1)
        acc = accuracy(y_hat, y)
        
        self.log("loss", loss)
        self.log("acc", acc)
        
        return {"loss": loss}

    def validation_step(self, val_batch, batch_idx):
        
        x, y = val_batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        
        _, y_hat = torch.max(logits, dim=1)
        acc = accuracy(y_hat, y)
        return {"val_step_loss": loss,"val_step_acc": acc}

    def validation_epoch_end(self, outputs):
        
        avg_loss = torch.stack([x["val_step_loss"] for x in outputs]).mean()
        self.log("val_loss", avg_loss)
        
        avg_acc = torch.stack([x["val_step_acc"] for x in outputs]).mean()
        self.log("val_acc", avg_acc)

    def test_step(self, test_batch, batch_idx):
        
        x, y = test_batch
        output = self.forward(x)
        _, y_hat = torch.max(output, dim=1)
        test_acc = accuracy(y_hat.cpu(), y.cpu())
        return {"test_acc": test_acc}

    def test_epoch_end(self, outputs):
        
        avg_test_acc = torch.stack([x["test_acc"] for x in outputs]).mean()
        self.log("avg_test_acc", avg_test_acc)

    def configure_optimizers(self):
        
        if self.args["optim"] == "sgd":
            self.optimizer = torch.optim.SGD(self.parameters(), lr=self.args["lr"], momentum=0.9)
        else:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.args["lr"])
        
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, 
                                                             max_lr=self.args["lr"], 
                                                             steps_per_epoch=self.args["steps_per_epoch"],
                                                             pct_start=0.4,
                                                             epochs=self.args["max_epochs"])
        self.scheduler = {"scheduler": self.scheduler, "interval" : "step" }
        
        
        return [self.optimizer], [self.scheduler]


if __name__ == "__main__":
    parser = ArgumentParser(description="PyTorch X-ray Example")

    # Early stopping parameters
    parser.add_argument(
        "--es_monitor", type=str, default="val_loss", help="Early stopping monitor parameter"
    )

    parser.add_argument("--es_mode", type=str, default="min", help="Early stopping mode parameter")

    parser.add_argument(
        "--es_verbose", type=bool, default=True, help="Early stopping verbose parameter"
    )

    parser.add_argument(
        "--es_patience", type=int, default=20, help="Early stopping patience parameter"
    )

    parser.add_argument(
        "--arch", type=str, default="cnn", help="Architecture to train (cnn or vit)"
    )

    parser.add_argument(
        "--pre_trained", type=str, default="yes", help="Should the model be pre-trained with ImageNet weights (default: yes)"
    )

    parser.add_argument(
        "--data_aug", type=str, default="yes", help="Should should data augmentation be used during training (default: yes)"
    )

    parser.add_argument(
        "--optim", type=str, default="adam", help="Which optimizer should be used (adam or sgd)"
    )

    parser.add_argument(
            "--batch_size", type=int, default=16, metavar="BS", help="input batch size for training (default: 16)",
    )
    parser.add_argument(
            "--num_workers", type=int, default=3, metavar="N", help="number of workers (default: 3)",
    )
    parser.add_argument(
            "--lr", type=float, default=1e-6, metavar="LR", help="learning rate (default: 1e-6)",
     )
    
    parser.add_argument(
            "--max_epochs", type=int, default=50, metavar="EPOCHS", help="Maximum number of epochs (default: 50)",
    )

    parser.add_argument(
            "--gpu", type=int, default=0, metavar="GPU", help="Specify which gpu to use (default: 0)",
    )

    mlflow.pytorch.autolog()

    args = parser.parse_args()
    dict_args = vars(args)

    if "accelerator" in dict_args:
        if dict_args["accelerator"] == "None":
            dict_args["accelerator"] = None

    dm = X_RayDataModule(**dict_args)
    dm.prepare_data()
    dm.setup(stage="fit")

    train_set_size = dm.train_set_size()
    batch_size = dict_args["batch_size"]
    steps_per_epoch = math.ceil( train_set_size/batch_size)
    dict_args["steps_per_epoch"] = steps_per_epoch

    model = LightningX_RayClassifier(**dict_args)


    early_stopping = EarlyStopping(
        monitor=dict_args["es_monitor"],
        mode=dict_args["es_mode"],
        verbose=dict_args["es_verbose"],
        patience=dict_args["es_patience"],
    )

    if dict_args["pre_trained"] == "yes":
        transfer = "pre-trained"
    else:
        transfer = "not-pre-trained"

    if dict_args["data_aug"] == "yes":
        aug = "data-augmentation"
    else:
        aug = "no-data-augmentation"

    run_name = str(dict_args["arch"]) + "_" + str(dict_args["optim"]) + "_" + str(dict_args["lr"]) + "_" + str(dict_args["batch_size"]) + "_" + transfer + "_" + aug

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.getcwd(), 
        save_top_k=1, 
        verbose=True, 
        monitor="val_loss", 
        mode="min", 
        prefix=run_name,
    )
    lr_logger = LearningRateMonitor()

    gpu = dict_args["gpu"]

    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[lr_logger, early_stopping], 
        checkpoint_callback=checkpoint_callback,
        gpus=[gpu],
        #auto_select_gpus=True,
        max_epochs=dict_args["max_epochs"],
        logger=pl.loggers.TensorBoardLogger('lightning_logs/', name=run_name),
        precision=16
    )

    with mlflow.start_run(run_name=run_name):
        trainer.fit(model, dm)
        trainer.test()
    
    #PATH='/nfs/student/e/ejguill/ece535/epoch=28-step=12817-best.ckpt'
    #model_from_check_point = LightningX_RayClassifier.load_from_checkpoint(PATH)
    #trainer.test(model_from_check_point,datamodule=dm)
