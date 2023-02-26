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

    

    #Resize, flip, convert, normalize images for training
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(size=224,scale=(0.3, 1.0)),
            #Flip image vertically and horizontally with prob 0.5
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        #Resize and crop images for validation
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        #Resize and crop images for testing (locked away)
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
        
        
    #Set metadata file and image data directory
    data_dir = '../HAM10000_images'




    metric = evaluate.load("accuracy")
    def compute_metrics(p):
        return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)
    
    
    train_ds = load_from_disk("../train_data.hf")
    val_ds = load_from_disk("../val_data.hf")
    test_ds = load_from_disk("../test_data.hf")
    
    

    class_names = train_ds.classes
    
    print(class_names)
    print(len())
    
    
    model_name_or_path = 'google/vit-base-patch16-224-in21k'
    
    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    
    print('HERE')
    
    """
    processed_data = {}
    for i in ['train', 'val', 'test']:
        print(i)
        data_list = []
        for j in range(len(image_datasets[i])):
            print(j)
            data_pt = {}
            data_pt['pixel_values'] = image_processor(image_datasets['train'][j][0], return_tensors='pt')['pixel_values']
            image_processor(image_datasets['train'][j][0], return_tensors='pt')
            data_pt['labels'] = image_datasets['train'][j][1]
            data_list += [data_pt]
        processed_data[i] = data_list
        """
    
    print('hH')
    
    
    def gen():
        for j in range(len(image_datasets['train'])):
            if (j%10 == 0):
                print(j)
            yield {"image": image_datasets['train'][j][0], "label": image_datasets['train'][j][1], "info": image_datasets['train'][j][2]}

    def val_gen():
        for j in range(len(image_datasets['val'])):
            if (j%10 == 0):
                print(j)
            yield {"image": image_datasets['val'][j][0], "label": image_datasets['val'][j][1], "info": image_datasets['val'][j][2]}

    def test_gen():
        for j in range(len(image_datasets['test'])):
            if (j%10 == 0):
                print(j)
            yield {"image": image_datasets['test'][j][0], "label": image_datasets['test'][j][1], "info": image_datasets['test'][j][2]}


    train_ds = Dataset.from_generator(gen)
    val_ds = Dataset.from_generator(val_gen)
    test_ds = Dataset.from_generator(test_gen)

    test_dataset.save_to_disk("./train_data.hf")
    
    
    
    
    
    
    
    
    
    
    
    
    
    ds = Dataset.from_dict(image_datasets['train'])
        
        
    def transform(example_batch):
        # Take a list of PIL images and turn them to pixel values
        inputs = image_processor([x for x in example_batch[0]], return_tensors='pt')
    
        # Don't forget to include the labels!
        inputs['labels'] = example_batch[1]
        return inputs

    prepared_ds = image_datasets.with_transform(transform)
    
    print('HERE2')
    
    model = ViTForImageClassification.from_pretrained(
        model_name_or_path,
        num_labels=len(class_names),
        id2label={str(i): c for i, c in enumerate(class_names)},
        label2id={c: str(i) for i, c in enumerate(class_names)}
    )
    
    for p in model.parameters():
        p.requires_grad = False
        
    # Turn back on the classifier weights
    for p in model.classifier.parameters():
        p.requires_grad=True
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    
    output_dir = "../model2_final"
    training_args = TrainingArguments(
        disable_tqdm=False,
        output_dir=output_dir,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        load_best_model_at_end=True,
        dataloader_num_workers=0,  
        gradient_accumulation_steps=8,
    )
    
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=image_datasets['train'],
        eval_dataset=image_datasets['val'],
        tokenizer=image_processor,
    )
    
    train_results = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()
    
    
    metrics = trainer.evaluate(image_datasets['val'])
    
    """
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    """
    

    """   
    # Get a batch of training data
    inputs, classes = next(iter(dataloaders['train']))
    print(inputs.shape)
    print(classes)
    
    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)
    print(out.shape)
    
    imshow(out, title=[class_names[x] for x in classes])
    """
    """
    # Load a pretrained model and reset final fully connected layer for this particular classification problem.
    
    image_processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
    model = SwinForImageClassification.from_pretrained(
        "microsoft/swin-tiny-patch4-window7-224",
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    
    #Print model info
    print("Num labels:", model.num_labels)
    print("\nModel config:", model.config)
    num_params, size_all_mb = get_model_info(model)
    print("Number of trainable params:", num_params)
    print('Model size: {:.3f}MB'.format(size_all_mb))
    
    #Freeze model
    for p in model.parameters():
        p.requires_grad = False
    
    # Turn back on the classifier weights
    for p in model.classifier.parameters():
        p.requires_grad=True
    
    # Ok now how many trainable parameters do we have?
    num_params, size_all_mb = get_model_info(model)
    print("Number of trainable params:", num_params)
    print('Model size: {:.3f}MB'.format(size_all_mb))
    


    training_args = TrainingArguments(
        disable_tqdm=False,
        output_dir=output_dir,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        load_best_model_at_end=True,
        dataloader_num_workers=0,  
        gradient_accumulation_steps=8,
    )
    
    base_learning_rate = 1e-3
    total_train_batch_size = (
        training_args.train_batch_size * training_args.gradient_accumulation_steps * training_args.world_size
    )
    
    training_args.learning_rate = base_learning_rate * total_train_batch_size / 256
    print("Set learning rate to:", training_args.learning_rate)
        
    metric = evaluate.load("accuracy")
    
        
    print("TRAINING\n\n")
    
    
    
    
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=image_datasets['train'],
        eval_dataset=image_datasets['val'],
        tokenizer=image_processor,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )
    
    # Train
    train_results = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()
    
    
    #Evaluate on test set
    #metrics = trainer.evaluate(image_datasets['test'])
    #trainer.log_metrics("eval", metrics)
    """
