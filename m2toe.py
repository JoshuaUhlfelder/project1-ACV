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
from tqdm import tqdm
import transformers
from transformers import (
    AutoImageProcessor, 
    ViTForImageClassification, 
    SwinForImageClassification,
    TrainingArguments, 
    Trainer,
    ResNetModel,
    AutoTokenizer, 
    BertModel,
    BertPreTrainedModel,
)
from transformers.modeling_outputs import SequenceClassifierOutput
import evaluate

cudnn.benchmark = True
plt.ion()


"""
********
m2toe.py
********

Single (or multi) data point classification on model 2

Set directory to image files, a metadata file organized
like the metadata file fom the HAM10000 dataset, and
and output directory for model files.

Function will output 


SET PARAMS BELOW
"""
data_dir = '../mytoe'
metadata = 'toe_metadata.csv'
output_dir = "../model2_toe"



#Dataset creation for train, val, test
class MyDataset(torch.utils.data.Dataset):
    """
    This class needs the directory where all the images are stored,
    a metadata file, the transform operations for each set,
    and a list of lesions in each set
    """
    def __init__(
        #Needs directory of images, metadata file, and transformations, 
        #and the lesion ids
        self,
        images_dir,
        metadata,
        image_transform,
        lesion_ids,
    ):
        self.lesion_ids = lesion_ids
        self.image_transform = image_transform
        
        # Retrieves class names form metadata file
        self.classes = self.get_class_names(metadata)
        
        # Assign a unique label index to each class name
        self.class_labels = {name: idx for idx, name in enumerate(self.classes)}
        
        # Collects the labels, images, and other information (age, sex) into 
        # a single list
        image_files, labels, info = self.get_image_filenames_with_labels(
            images_dir,
            self.class_labels,
            metadata,
            self.lesion_ids,
        )
        
        # Avoid memory leaks - put into np arrays
        self.image_files = np.array(image_files)
        self.labels = np.array(labels).astype("int")
        
        # Set size of dataset
        self.num_images = len(self.image_files)
        
    def __len__(self):
        return self.num_images
        
    def get_class_names(self, metadata):
        #Return all classes as list of strings by iterating through metadata
        #Takes from the original dataset
        class_names = set()
        with open("./HAM10000_metadata.csv", newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            next(reader)
            for row in reader:
                class_names.add(row[2])
        
        return sorted(list(class_names)) #convert set to list and return
    
    #The images are organized , all ending with .jpg, and with uniform naming structure
    def get_image_filenames_with_labels(self, images_dir, class_labels, metadata, lesion_ids):
        image_files = []
        labels = []
        info = []
        
        #Iterate over all lesions in the specific set
        all_data = pd.read_csv(metadata)
        for lesion_id in lesion_ids:
            results = all_data.loc[all_data['lesion_id'] == lesion_id]
            
            #For each image of the lesion
            #add the image name, label, and demographic info to the lists
            results = results.reset_index()
            for index, row in results.iterrows():
                image_files += [images_dir + '/' + (row['image_id'] + '.jpg')]
                labels += [class_labels[row['dx']]]
                info += (row['lesion_id'], row['dx_type'], row['age'], 
                         row['sex'], row['localization'],)
        return image_files, labels, info
    
    def __getitem__(self, idx):  
        # Retrieve an image from the list, load it, transform it, 
        # and return it along with its label
        #Bad data returned as None
        try:
            # Try to open the image file and convert it to RGB.
            image = Image.open(self.image_files[idx]).convert('RGB')
            label = self.labels[idx]
            
            #Will use this in future models
            #info = self.info[idx]
            
            # Apply the image transform
            image = self.image_transform(image)
            
            return image, label
        except Exception as exc:  # <--- i know this isn't the best exception handling
            return None



#Resize, flip, convert, normalize images for training
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=224,scale=(0.2, 1.0)),
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
    



# Returns the lesion IDs for each split
def train_val_test_split(metadata):
    
    #Read in metadata and separate lesions (not images) into training, test, and validation groups
    #Train = 60% of lesions
    #Val = 20% of lesions
    #Test = 20% of lesions
    df = pd.read_csv(metadata)
    train_les = df['lesion_id'].drop_duplicates()
    return {'test': train_les}





#SPut data into test df
splits = train_val_test_split(metadata)

#create test dataset from input
print("Setting up datasets")
image_datasets = {x: MyDataset(data_dir, metadata, data_transforms[x], sorted(list(splits[x]))) for x in ['test']}


dataset_sizes = {x: len(image_datasets[x]) for x in ['test']}
class_names = image_datasets['test'].classes
num_classes = len(class_names)
print(dataset_sizes)
print(class_names)

  
def collate_fn(batch):
    # Filter failed images first
    #batch = list(filter(lambda x: x is not None, batch))
    
    # Now collate into mini-batches
    return {
        "pixel_values": torch.stack([x[0] for x in batch]),
        "labels": torch.LongTensor([x[1] for x in batch]),
    }


# From Austin Reiter
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

# Create a lookup table to go between label name and index
id2label = {}
label2id = {}
for idx, label in enumerate(class_names):
    id2label[str(idx)] = label
    label2id[label] = str(idx)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)



# Load the best model from the directory below into 'model'
#Takes the image processor from 21k ImageNet
image_processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained("../model2_final")

#Print out model info
print(model.classifier)
print("Num labels:", model.num_labels)
print("\nModel config:", model.config)

# Print out model size
num_params, size_all_mb = get_model_info(model)
print("Number of trainable params:", num_params)
print('Model size: {:.3f}MB'.format(size_all_mb))

#Set args for training
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=9,
    lr_scheduler_type='cosine',
    logging_steps=10,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
    load_best_model_at_end=True,
    dataloader_num_workers=0,  
    #gradient_accumulation_steps=8,
)

#Modify lr
base_learning_rate = 1e-3
total_train_batch_size = (
    training_args.train_batch_size * training_args.gradient_accumulation_steps * training_args.world_size
)

training_args.learning_rate = base_learning_rate * total_train_batch_size / 256
print("Set learning rate to:", training_args.learning_rate)
    
#Use accuracy as metric for evaluation
metric = evaluate.load("accuracy")
def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)
    
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=image_datasets['test'],
    eval_dataset=image_datasets['test'],
    tokenizer=image_processor,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)


print('predicting')
predictions = trainer.predict(image_datasets['test'])
print("Result:", predictions.predictions)
preds = np.argmax(predictions.predictions, axis=-1)
trues = predictions.label_ids
from sklearn.metrics import recall_score, precision_score

#print predictions
print(preds)
print(trues)

#Get recall and precision
print(recall_score(trues, preds, average=None))
print(precision_score(trues, preds, average=None))

