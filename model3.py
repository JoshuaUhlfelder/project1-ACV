#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 01:19:44 2023

@author: joshuauhlfelder
"""

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


        
#Set metadata file and image data directory
data_dir = '../HAM10000_images'
metadata = 'HAM10000_metadata.csv'

#Set directory to output model
output_dir = "../model3_final"


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
        self.info = np.array(info).astype("object")
        
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
                info.append(tuple((row['age'], row['sex'], row['localization'],)))
        return image_files, labels, info
    
    def __getitem__(self, idx):  
        # Retrieve an image from the list, load it, transform it, 
        # and return it along with its label
        #Bad data returned as None
        try:
            # Try to open the image file and convert it to RGB.
            image = Image.open(self.image_files[idx]).convert('RGB')
            label = self.labels[idx]
            info = self.info[idx]
            
            #Will used this in future models
            #info = self.info[idx]
            
            # Apply the image transform
            image = self.image_transform(image)
            
            return image, label, info
        #Bad images return None
        except Exception as exc:  
            return None

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





#Split lesions into sets
splits = train_val_test_split(metadata)

#create datasets
print("Setting up datasets")
image_datasets = {x: MyDataset(data_dir, metadata, data_transforms[x], sorted(list(splits[x]))) for x in ['train','val','test']}

#print("Setting up dataloaders")
#dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=8,
#                                             shuffle=True, num_workers=0, collate_fn=collate_fn)
#              for x in ['train', 'val']}


dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
class_names = image_datasets['train'].classes
num_classes = len(class_names)
print(dataset_sizes)
print(class_names)

    

class MyTokenizer():
    """
    This class needs the directory where all the images are stored,
    a metadata file, the transform operations for each set,
    and a list of lesions in each set
    """
    def __init__(
        self,
        metadata, #metadata file to get localizations
    ):
        # Retrieves localization list from metadata file
        self.localizations = self.get_localizations(metadata)

    
    def get_localizations(self, metadata):
        #Return a dictionary of all localizations mapped to unique indexes
        l_names = set()
        with open(metadata, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            next(reader)
            for row in reader:
                l_names.add(row[6])
        l_names = sorted(l_names)
        localizations = {}
        for idx, label in enumerate(l_names):
            localizations[label] = idx
        
        return localizations #convert set to list and return
    
    def tokenize(self, input_tuples):
        length = len(input_tuples)
        ids = []
        types = []
        mask = []
        #For each frame in the batch, put the sex, localization, and age into a list
        for i in range(length):
            input_tuple = tuple(input_tuples[i])
            age = input_tuple[0]
            sex = input_tuple[1]
            loc = input_tuple[2]
            
            
            #Set sex to label
            #0 = male, 1 = female, #-1 = unknown
            if sex == 'male':
                new_sex = 0
            elif sex == 'female':
                new_sex = 1
            else:
                sex == 2
                   
            #Get the localization label from table in tokenizer
            try:
                new_loc = self.localizations[loc]
            except:
                raise Exception("Error finding localization")
            
            #Check if age field is unknown and set to 0
            if age == 'nan':
                age = 0
            
            #Add each list to end of prev. list
            ids.append([int(float(age)), int(new_sex), int(new_loc)])
            types.append([0,0,0])
            mask.append([1,1,1])
            
        #Convert the lists of lists to tensors
        ids = torch.tensor(ids)
        types = torch.tensor(types)
        mask = torch.tensor(mask)
            
        
        return {'input_ids': ids, 'token_type_ids': types, 'attention_mask': mask}
        

image_preprocessor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
text_tokenizer = MyTokenizer(metadata)


def collate_fn(batch):
    # Filter failed images first
    
    tokenized_text = text_tokenizer.tokenize([x[2] for x in batch])

    # Process the images
    processed_images = image_preprocessor([x[0] for x in batch], return_tensors="pt", padding=True)
    
    # Collect the labels
    labels = torch.LongTensor([x[1] for x in batch])
    
    return {
        "text": tokenized_text,
        "images": processed_images,
        "labels": labels,
    }



dataloader = torch.utils.data.DataLoader(
    image_datasets['train'], 
    batch_size=2,
    shuffle=True, 
    num_workers=0, 
    collate_fn=collate_fn,
)



for batch in dataloader:
    break


batch["labels"]


batch["text"]['attention_mask']

batch["images"]["pixel_values"].shape





#Multimodal Bert
"""
From Austin Reiter -
 ---huggingface_examples.py---
- with some modifications to fit data sizes and 
addition of custom resnet50
"""
class MultimodalBertClassifier(nn.Module):
    def __init__(
        self,
        num_labels,
    ):  
        super().__init__()
        
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.resnet = ResNetModel.from_pretrained("microsoft/resnet-50")
        
        self.num_labels = num_labels
        
        # FC layer to project image embeddings to hidden dims of BERT.
        # We hard-code the assumption that the output of the resnet last 
        # hidden state is 2048 feature dims.
        self.image_tokenizer = nn.Linear(2048, self.bert.config.hidden_size)
        
        # Image position embeddings.  We hard-code assumption that output of 
        # last hidden state in resnet model is [batch_size, 2048, 7, 7].  The
        # 7*7 gets flattened to 49 sequence length.
        self.image_pos_emb = nn.Embedding(49, self.bert.config.hidden_size)
        
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size, self.num_labels)
        
    def forward(
        self,
        text,
        images,
        labels=None,
    ):
        # Encode the images.  The last hidden state (which is what we want)
        # has a shape of: [batch_size, 2048, 7, 7].
        image_outputs = self.resnet(**images)
        
        # Permute the dimensions and project to hidden dims for BERT.  Be sure
        # to flatten the spatial dims out first:
        #
        # [batch_size, 2048, 7, 7] -> [batch_size, 2048, 49] -> [batch_size, 49, 2048]
        image_emb = image_outputs.last_hidden_state.flatten(2).permute(0, 2, 1)
        image_emb = self.image_tokenizer(image_emb)  # [batch_size, 49, 2048] -> [batch_size, 49, 768]
        
        # Apply position embeddings and token-type embeddings to the image embeddings.
        # Note that we use different position embeddings for the image tokens
        # from the text tokens.
        image_position_ids = torch.arange(image_emb.shape[1]).repeat(image_emb.shape[0], 1).to(image_emb.device)
        image_position_emb = self.image_pos_emb(image_position_ids)
        
        # Also, use the token_type_id=1 for images, and 0 is used for all the text.
        # Use the pre-trained BERT model to get token-type embeddings
        image_type_ids = torch.LongTensor([1] * image_emb.shape[1]).repeat(image_emb.shape[0], 1).to(image_emb.device)
        image_type_emb = self.bert.embeddings.token_type_embeddings(image_type_ids)
        
        # Now sum them all up and normalize
        image_emb = image_emb + image_position_emb + image_type_emb
        image_emb = self.bert.embeddings.LayerNorm(image_emb)
        image_emb = self.bert.embeddings.dropout(image_emb)
        
        # Embed the text, add positional embeddings and store the embedding outputs
        text_embedding_output = self.bert.embeddings(
            input_ids=text['input_ids'],
            token_type_ids=text['token_type_ids'],
        )
        
        # Concatenate all of the embeddings on the time dimension
        embedding_output = torch.cat([text_embedding_output, image_emb], 1)
        
        # Before we put all this into the transformer encoder, we need to 
        # extend the attention mask to include all of the image token embeddings, but
        # exclude any of the padded text token embeddings.  In Huggingface notation,
        # 1 means keep and 0 means ignore
        image_attention_mask = torch.LongTensor([1] * image_emb.shape[1]).repeat(image_emb.shape[0], 1).to(image_emb.device)
        extended_attention_mask = torch.cat([text["attention_mask"], image_attention_mask], 1)
        
        # Make broadcastable attention masks so that masked tokens are ignored (does some pre-processing
        # to prepare for the encoder)
        input_shape = (embedding_output.shape[0], embedding_output.shape[1])
        extended_attention_mask = self.bert.get_extended_attention_mask(extended_attention_mask, input_shape)

        # And then encode
        encoder_outputs = self.bert.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
        )
        
        # Get the pooled output for classification and apply the classifier head
        sequence_outputs = encoder_outputs[0]
        pooled_output = self.bert.pooler(sequence_outputs)  # Use CLS_TOKEN embedding
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )




# Quick check if the forward inferencing works
model = MultimodalBertClassifier(num_labels=7)

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
    


# Print model info
num_params, size_all_mb = get_model_info(model)

print("Number of trainable params:", num_params)
print('Model size: {:.3f}MB'.format(size_all_mb))




# Create a lookup table to go between label name and index
id2label = {}
label2id = {}
for idx, label in enumerate(class_names):
    id2label[str(idx)] = label
    label2id[label] = str(idx)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
    load_best_model_at_end=True,
    dataloader_num_workers=0,  
    gradient_accumulation_steps=4,
)




# Compute absolute learning rate
base_learning_rate = 1e-3
total_train_batch_size = (
    training_args.train_batch_size * training_args.gradient_accumulation_steps * training_args.world_size
)

training_args.learning_rate = base_learning_rate * total_train_batch_size / 256
print("Set learning rate to:", training_args.learning_rate)


metric = evaluate.load("accuracy")
def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)



# Create the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=image_datasets["train"],
    eval_dataset=image_datasets["val"],
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)



# Train
train_results = trainer.train()
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()
