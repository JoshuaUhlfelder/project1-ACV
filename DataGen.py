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
from datasets import Dataset, Image as im
from tqdm import tqdm

from transformers import (
    AutoImageProcessor, 
    ViTForImageClassification, 
    SwinForImageClassification,
    TrainingArguments, 
    Trainer,
    ViTFeatureExtractor,
)
import evaluate



data_dir = '../HAM10000_images'
metadata = 'HAM10000_metadata.csv'


def train_val_test_split(metadata):
    
    #Read in metadata and separate lesions (not images) into training, test, and validation groups
    #Train = 60% of lesions
    #Val = 20% of lesions
    #Test = 20% of lesions
    df = pd.read_csv(metadata)
    train_les, eval_les = train_test_split(df['lesion_id'].drop_duplicates(), test_size=0.4, random_state=0)
    val_les, test_les = train_test_split(eval_les, test_size=0.5, random_state=0)
    
    return {'train': train_les, 'val': val_les, 'test': test_les}

splits = train_val_test_split(metadata)

class GenDataSet():
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
        lesion_ids,
    ):
        self.lesion_ids = lesion_ids
        
        # Retrieves class names form metadata file
        self.classes = self.get_class_names(metadata)
        
        # Assign a unique label index to each class name
        self.class_labels = {name: idx for idx, name in enumerate(self.classes)}
        
        # Next, let's collect all image files underneath each class name directory 
        # as a single list of image files.  We need to correspond the class label
        # to each image.
        image_files, labels, lesion, dx_type, age, sex, localization = self.get_image_filenames_with_labels(
            images_dir,
            self.class_labels,
            metadata,
            self.lesion_ids,
        )
        
        # This is a trick to avoid memory leaks over very large datasets.
        self.image_files = np.array(image_files)
        self.labels = np.array(labels).astype("int")
        self.lesion = np.array(lesion).astype("str")
        self.dx_type = np.array(dx_type).astype("str")
        self.age = np.array(age).astype("float")
        self.sex = np.array(sex).astype("str")
        self.localization = np.array(localization).astype("str")
        
        # How many total images do we need to iterate in this entire dataset?
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
        lesion =[]
        dx_type =[]
        age = []
        sex = []
        localization = []
        
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
                lesion += [row['lesion_id']]
                dx_type += [row['dx_type']]
                age += [row['age']]
                sex += [row['sex']]
                localization += [row['localization']]
        return image_files, labels, lesion, dx_type, age, sex, localization
    
    def __getitem__(self, idx):
        # Retrieve an image from the list, load it, transform it, 
        # and return it along with its label
        #Bad data returned as None
        try:
            # Try to open the image file and convert it to RGB.
            image = Image.open(self.image_files[idx]).convert('RGB')
            image = feature.encode_example(image)
            label = self.labels[idx]
            lesion = self.lesion[idx]
            dx_type = self.dx_type[idx]
            age = self.age[idx]
            sex = self.sex[idx]
            localization = self.localization[idx]
            
            return {'image':image, 'label':label, 'lesion':lesion, 'dx_type':dx_type, 'age':age, 'sex':sex, 'localization':localization}
        except Exception as exc:  # <--- i know this isn't the best exception handling
            return None


image_datasets = {x: GenDataSet(data_dir, metadata, sorted(list(splits[x]))) for x in ['train','val','test']}

feature = im()




def train_gen():
    yield {"image": image_datasets['train'][0]["image"], 
           "label": image_datasets['train'][0]["label"],
            "lesion": image_datasets['train'][0]["lesion"],
            "dx_type": image_datasets['train'][0]["dx_type"],
            "age": image_datasets['train'][0]["age"],
            "sex": image_datasets['train'][0]["sex"],
            "localization": image_datasets['train'][0]["localization"]}

def val_gen():
    yield {"image": image_datasets['val'][0]["image"], 
           "label": image_datasets['val'][0]["label"],
            "lesion": image_datasets['val'][0]["lesion"],
            "dx_type": image_datasets['val'][0]["dx_type"],
            "age": image_datasets['val'][0]["age"],
            "sex": image_datasets['val'][0]["sex"],
            "localization": image_datasets['val'][0]["localization"]}
def test_gen():
    yield {"image": image_datasets['test'][0]["image"], 
           "label": image_datasets['test'][0]["label"],
            "lesion": image_datasets['test'][0]["lesion"],
            "dx_type": image_datasets['test'][0]["dx_type"],
            "age": image_datasets['test'][0]["age"],
            "sex": image_datasets['test'][0]["sex"],
            "localization": image_datasets['test'][0]["localization"]}


print("Compiling Training")
train_ds = Dataset.from_generator(train_gen)
for j in tqdm(range(1,len(image_datasets['train']))):
    train_ds= train_ds.add_item(image_datasets['train'][j])

train_ds.save_to_disk("../train_data.hf")
print("Compiling Validation")
val_ds = Dataset.from_generator(val_gen)
for j in tqdm(range(1,len(image_datasets['val']))):
    val_ds = val_ds.add_item(image_datasets['val'][j])
    
val_ds.save_to_disk("../val_data.hf")

print("Compiling Test")
test_ds = Dataset.from_generator(test_gen)
for j in tqdm(range(1,len(image_datasets['test']))):
    test_ds = test_ds.add_item(image_datasets['test'][j])
    
test_ds.save_to_disk("../test_data.hf")
