#!/usr/bin/env python
# coding: utf-8


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import pandas as pd
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from PIL import Image
import csv
from sklearn.model_selection import train_test_split
from datasets import Dataset
from tqdm import tqdm
from datasets import load_dataset
from datasets import load_from_disk
import io

from transformers import (
    AutoImageProcessor, 
    ViTForImageClassification, 
    SwinForImageClassification,
    TrainingArguments, 
    Trainer,
    ViTFeatureExtractor,
)
import evaluate

cudnn.benchmark = True
plt.ion()


def collate_fn(batch):
    # Filter failed images first
    #batch = list(filter(lambda x: x is not None, batch))
    
    # Now collate into mini-batches
    return {
        "pixel_values": torch.stack([x[0] for x in batch]),
        "labels": torch.tensor([x[1] for x in batch]),
    }



# A useful function to see the size and # of params of a model
def get_model_info(model):
    # Compute number of trainable parameters in the model
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Compute the size of the model in MB
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        
    size_all_mb = (param_size + buffer_size) / 1024**2
    
    return num_params, size_all_mb



if __name__ == "__main__":

    
    #Set image data directory
    data_dir = '../HAM10000_images'

    #Resize, flip, convert, normalize images for training
    """
    train_transform = transforms.Compose([
        Image.frombytes(mode, size, data)
        transforms.RandomResizedCrop(size=224,scale=(0.3, 1.0)),
        #Flip image vertically and horizontally with prob 0.5
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    """
    #Load in all datasets
    #Can only be used after running DataGen.py
    train_ds = load_from_disk("../train_data.hf")
    val_ds = load_from_disk("../val_data.hf")
    test_ds = load_from_disk("../test_data.hf")
    
    """
    train_ds.set_transform(transforms)
    val_ds.set_transform(val_transform)
    test_ds.set_transform(val_transform)
    """
    
    def transforms(batch):
        byts = batch['image']['bytes']
        batch["pixel_values"] = Image.open(io.BytesIO(byts))
        return batch
    
    
    train_ds = tqdm(train_ds.map(transforms, remove_columns=["image"], 
                                 batched=False, keep_in_memory=True, writer_batch_size=500))
    
    
    #train_ds[0:8]['image']
    
    
    img = train_ds[0]['image']['bytes']
    image = io.BytesIO(img)
    image = Image.open(image)
    
    
    print(train_ds[0]['image'])
    
    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    