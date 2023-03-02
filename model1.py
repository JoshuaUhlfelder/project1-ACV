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
model1.py
********

Trains a ResNet50 on the training data

Set directory to image files and a metadata file organized
like the metadata file fom the HAM10000 dataset

Function will output an evaluation of the validation data


SET PARAMS BELOW
"""
#Set metadata file and image data directory
data_dir = '../HAM10000_images'
metadata = 'HAM10000_metadata.csv'


#Dataset creation for train, val, test
class MyDataset(torch.utils.data.Dataset):
    """
    This class needs the directory where all the images are stored,
    a metadata file, the transform operations for each set,
    and a list of lesions in each set
    """
    def __init__(
        #Needs directory of images, metadata file, and transformations
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
        
        # Collect images, info, and labels into lists
        image_files, labels, info = self.get_image_filenames_with_labels(
            images_dir,
            self.class_labels,
            metadata,
            self.lesion_ids,
        )
        
        # Convert to np arrays
        self.image_files = np.array(image_files)
        self.labels = np.array(labels).astype("int")
        
        # Size of datasets
        self.num_images = len(self.image_files)
        
    def __len__(self):
        return self.num_images
        
    def get_class_names(self, metadata):
        #Return all classes as list of strings by iterating through metadata
        class_names = set()
        with open(metadata, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            next(reader)
            for row in reader:
                class_names.add(row[2])
        
        return sorted(list(class_names)) #convert set to list and return
    
    #The images are organized cleanly, all ending with .jpg, and with uniform naming structure
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

# Returns the lesion IDs for each split
def train_val_test_split(metadata):
    
    #Read in metadata and separate lesions (not images) into training, test, and validation groups
    #Train = 60% of lesions
    #Val = 20% of lesions
    #Test = 20% of lesions
    df = pd.read_csv(metadata)
    train_les, eval_les = train_test_split(df['lesion_id'].drop_duplicates(), test_size=0.4, random_state=0)
    val_les, test_les = train_test_split(eval_les, test_size=0.5, random_state=0)
    
    return {'train': train_les, 'val': val_les, 'test': test_les}





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

#Split lesions into sets
splits = train_val_test_split(metadata)

#create datasets
print("Setting up datasets")
image_datasets = {x: MyDataset(data_dir, metadata, data_transforms[x], sorted(list(splits[x]))) for x in ['train','val','test']}


def collate_fn(batch):
    # Filter failed images first
    batch = list(filter(lambda x: x is not None, batch))
    
    # Now collate into mini-batches
    images = torch.stack([b[0] for b in batch])
    labels = torch.LongTensor([b[1] for b in batch])
    
    return images, labels



#creat dataloaders - set batch size here
print("Setting up dataloaders")
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=8,
                                             shuffle=True, num_workers=0, collate_fn=collate_fn)
              for x in ['train', 'val']}


#Print our classes
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
class_names = image_datasets['train'].classes
print(dataset_sizes)
print(class_names)

# to correct device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

  

def train_model(model, criterion, optimizer, scheduler, num_epochs=10):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        print(scheduler.get_last_lr())

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

# Load a pretrained model and reset final fully connected layer for this particular classification problem.

model_ft = models.resnet50(weights="IMAGENET1K_V1")
#model_ft = models.vgg19_bn(weights="IMAGENET1K_V1")



num_ftrs = model_ft.fc.in_features

# Add linear classification layer
model_ft.fc = nn.Linear(num_ftrs, 7)

# Move the model to the correct device
model_ft = model_ft.to(device)


# Using cross-entropy
criterion = nn.CrossEntropyLoss()

# Setup the optimizer to update the model parameters
optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.00005)


# Decay LR by a factor of 0.8 after a linear warmup
scheduler1 = lr_scheduler.LinearLR(optimizer_ft, start_factor=0.05, total_iters=4)
scheduler2 = lr_scheduler.ExponentialLR(optimizer_ft, gamma=0.8)
scheduler = lr_scheduler.SequentialLR(optimizer_ft, 
                                      schedulers=[scheduler1, scheduler2], milestones=[3])



# Train and evaluate.  
model_ft = train_model(model_ft, criterion, optimizer_ft, scheduler,
                       num_epochs=15)



torch.save(model_ft.state_dict(), '../model1_final')



