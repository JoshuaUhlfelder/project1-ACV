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
cudnn.benchmark = True
plt.ion()

"""
********
m1toe.py
********

Single (or multi) data point classification on model 1

Set directory to image files, a metadata file organized
like the metadata file fom the HAM10000 dataset, and
and output directory for model files.

Function will output predictions


SET PARAMS BELOW
"""
data_dir = '../mytoe'
metadata = 'toe_metadata.csv'
output_dir = "../model1_toe"





#Put datapoints into dataset
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
        #Return none for bad images
        except Exception as exc:
            return None


# Returns the lesion IDs for each split
def train_val_test_split(metadata):
    
    #Read in metadata and separate lesions (not images) into test group
    df = pd.read_csv(metadata)
    train_les = df['lesion_id'].drop_duplicates()
    return {'test': train_les}



#Resize, flip, convert, normalize images for training
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=224,scale=(0.1, 1.0)),
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

    

#Put everything into a test df
splits = train_val_test_split(metadata)


#create datasets from organized data
print("Setting up datasets")
image_datasets = {x: MyDataset(data_dir, metadata, data_transforms[x], sorted(list(splits[x]))) for x in ['test']}




def collate_fn(batch):
    # Filter failed images
    batch = list(filter(lambda x: x is not None, batch))
    
    # Collate batches
    images = torch.stack([b[0] for b in batch])
    labels = torch.LongTensor([b[1] for b in batch])
    
    return images, labels


#Create dataloaders - only load one at a time for test
print("Setting up dataloaders")
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1,
                                             shuffle=True, num_workers=0, collate_fn=collate_fn)
              for x in ['test']}




#Not operating on cuda - unnecessary 
device = torch.device("cpu")
print(device)

 

# Load a pretrained model and reset final fully connected layer for this particular classification problem.
model = models.resnet50()# we do not specify pretrained, loading best model
num_ftrs = model.fc.in_features

# Add linear layer for classification
model.fc = nn.Linear(num_ftrs, 7)

# Move the model to the correct device (cpu)
model = model.to(device)

#Load in best model so far - set directory to save
model.load_state_dict(torch.load('../model1_final'))

all_labels = torch.tensor([])
all_preds = torch.tensor([])
#For every test case, add to a list of predicted and expected labels
for inputs, labels in tqdm(dataloaders['test']):
    inputs = inputs.to(device)
    labels = labels.to(device)
    
    all_labels = torch.cat((all_labels, labels))
    out = model(inputs)
    print("Result:", out)
    pre = torch.argmax(out, dim=1)
    all_preds = torch.cat((all_preds, pre))
#Convert the tensors to lists for sklearn
trues = all_labels.tolist()
preds = all_preds.tolist()

from sklearn.metrics import recall_score, precision_score

print(preds)
print(trues)

#Get recall and precision
print(recall_score(trues, preds, average=None))
print(precision_score(trues, preds, average=None))






