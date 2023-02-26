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
#from tqdm import tqdm

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

os.environ["TOKENIZERS_PARALLELISM"] = "false"


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
        
        # Next, let's collect all image files underneath each class name directory 
        # as a single list of image files.  We need to correspond the class label
        # to each image.
        image_files, labels, info = self.get_image_filenames_with_labels(
            images_dir,
            self.class_labels,
            metadata,
            self.lesion_ids,
        )
        
        # This is a trick to avoid memory leaks over very large datasets.
        self.image_files = np.array(image_files)
        self.labels = np.array(labels).astype("int")
        
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
            
            #Will used this in future models
            #info = self.info[idx]
            
            # Apply the image transform
            image = self.image_transform(image)
            
            return image, label
        except Exception as exc:  # <--- i know this isn't the best exception handling
            return None



def collate_fn(batch):
    # Filter failed images first
    #batch = list(filter(lambda x: x is not None, batch))
    
    # Now collate into mini-batches
    return {
        "pixel_values": torch.stack([x[0] for x in batch]),
        "labels": torch.tensor([x[1] for x in batch]),
    }


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
    metadata = 'HAM10000_metadata.csv'


    #Split lesions into sets
    splits = train_val_test_split(metadata)
    
    #create datasets
    print("Setting up datasets")
    image_datasets = {x: MyDataset(data_dir, metadata, data_transforms[x], sorted(list(splits[x]))) for x in ['train','val','test']}
    

    print(image_datasets['train'][0][0].shape)
    
    metric = evaluate.load("accuracy")
    def compute_metrics(p):
        return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)
    
    
    
    
    
    
    
    
    
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    class_names = image_datasets['train'].classes
    print(dataset_sizes)
    print(class_names)
    
    
    model_name_or_path = 'google/vit-base-patch16-224-in21k'
    feature_extractor = ViTFeatureExtractor.from_pretrained(model_name_or_path)
    
    model = ViTForImageClassification.from_pretrained(
        model_name_or_path,
        num_labels=len(class_names),
        id2label={str(i): c for i, c in enumerate(class_names)},
        label2id={c: str(i) for i, c in enumerate(class_names)}
    )
    
    
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
        train_dataset=image_datasets["train"],
        eval_dataset=image_datasets["val"],
        tokenizer=feature_extractor,
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