#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import json
import glob 
import itertools
from PIL import Image

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
plt.ion()   # interactive mode


# In[ ]:


# For straightforward datasets, sometimes you can make do with built-in PyTorch dataset objects.
# We want to apply automated data augmentations, which will be different for the training
# and eval scenarios

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

"""
# In[ ]:


data_dir = '../hymenoptera_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}

class_names = image_datasets['train'].classes
num_classes = len(class_names)


# In[ ]:


# Create a lookup table to go between label name and index
id2label = {}
label2id = {}
for idx, label in enumerate(class_names):
    id2label[str(idx)] = label
    label2id[label] = str(idx)


# In[ ]:


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x[0] for x in batch]),
        "labels": torch.LongTensor([x[1] for x in batch]),
    }


# In[ ]:


# Create a ViT pre-trained classifier to fine-tune
# (see more docs here:  https://huggingface.co/docs/transformers/model_doc/vit)
#
# We have a different number of classes than the way it was pre-trained, and so
# we need to modify that here.
image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=num_classes,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
).to('cuda:0')


# In[ ]:


# Check if our new classifier head got in there correctly
model.classifier


# In[ ]:


# And the new config parameters?
print("Num labels:", model.num_labels)
print("\nModel config:", model.config)


# In[ ]:



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


# In[ ]:


# Print model info
num_params, size_all_mb = get_model_info(model)

print("Number of trainable params:", num_params)
print('Model size: {:.3f}MB'.format(size_all_mb))


# In[ ]:


# Ok this is way too big for my poor little CPU.  Let's freeze the trunk and
# only train the classifier head.  First, freeze the entire model:
for p in model.parameters():
    p.requires_grad = False
    
# Turn back on the classifier weights
for p in model.classifier.parameters():
    p.requires_grad=True
    
# Ok now how many trainable parameters do we have?
num_params, size_all_mb = get_model_info(model)
print("Number of trainable params:", num_params)
print('Model size: {:.3f}MB'.format(size_all_mb))


# In[ ]:


# Setup the training arguments
output_dir = "./finetune_vit"

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=32,
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
#     gradient_accumulation_steps=8,
)


# In[ ]:


# Compute absolute learning rate
base_learning_rate = 1e-3
total_train_batch_size = (
    training_args.train_batch_size * training_args.gradient_accumulation_steps * training_args.world_size
)

training_args.learning_rate = base_learning_rate * total_train_batch_size / 256
print("Set learning rate to:", training_args.learning_rate)
"""

# In[ ]:


# Setup a function to compute accuracy metrics
metric = evaluate.load("accuracy")
def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)
"""

# In[ ]:


# Create the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=image_datasets['train'],
    eval_dataset=image_datasets['val'],
    tokenizer=image_processor,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)


# In[ ]:


# Train
train_results = trainer.train()
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()


# In[ ]:


# How did we do on the eval set?  We also see this progress during training, but if we had a 
# separate left out test set, its 1 line of code to evaluate on it!
metrics = trainer.evaluate(image_datasets['val'])
trainer.log_metrics("eval", metrics)


# In[ ]:


# Creating a model from a different pre-trained trunk is super easy.  Let's
# do an MAE pre-trained trunk
image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
model = ViTForImageClassification.from_pretrained(
    "facebook/vit-mae-base",
    num_labels=num_classes,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)


# In[ ]:


# How about the Swin Transformer?
image_processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
model = SwinForImageClassification.from_pretrained(
    "microsoft/swin-tiny-patch4-window7-224",
    num_labels=num_classes,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)

"""












# ## Let's now modify a BERT model to do multi-model classification on the Hateful Memes dataset.

# In[ ]:


class HatefulMemesDataset(torch.utils.data.Dataset):
    def __init__(self, json_file, images_dir):
        image_files = []
        captions = []
        labels = []
        with open(json_file, 'r') as f:
            json_list = list(f)

        for json_str in json_list:
            sample = json.loads(json_str)
            image_files.append(os.path.join(images_dir, sample['img']))
            captions.append(str(sample['text']))
            labels.append(int(sample['label']))
            
        self.image_files = np.array(image_files)
        self.captions = np.array(captions)
        self.labels = np.array(labels)
        
        self.num_samples = len(self.image_files)
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return {
            "image": Image.open(self.image_files[idx]),
            "caption": self.captions[idx],
            "label": self.labels[idx],
        }


# In[ ]:


# Create the text tokenizer and image pre-processors
text_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
image_preprocessor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")


# In[ ]:


def collate_fn(batch):
    # Tokenize the text and pad as necessary
    tokenized_text = text_tokenizer([x["caption"] for x in batch], return_tensors="pt", padding=True)
    
    # Process the images
    processed_images = image_preprocessor([x["image"] for x in batch], return_tensors="pt", padding=True)
    
    # Collect the labels
    labels = torch.LongTensor([x["label"] for x in batch])
    
    return {
        "text": tokenized_text,
        "images": processed_images,
        "labels": labels,
    }


# In[ ]:


# Create the train/eval/test datasets
train_dataset = HatefulMemesDataset('./data/train.jsonl', './data')
eval_dataset = HatefulMemesDataset('./data/dev.jsonl', './data')
# test_dataset = HatefulMemesDataset('./data/test.jsonl', './data')


# In[ ]:


# Test the dataset within a dataloader
dataloader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=32,
    shuffle=True, 
    num_workers=0, 
    collate_fn=collate_fn,
)


# In[ ]:


for batch in dataloader:
    break


# In[ ]:


batch["labels"]


# In[ ]:


batch["text"]['attention_mask']


# In[ ]:


batch["images"]["pixel_values"].shape


# In[ ]:


# Create a custom multimodal model by modifying BERT and using ResNet50
# to do multimodal classification
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
            input_ids=text.input_ids,
            token_type_ids=text.token_type_ids,
        )
        
        # Concatenate all of the embeddings on the time dimension
        embedding_output = torch.cat([text_embedding_output, image_emb], 1)
        
        # Before we put all this into the transformer encoder, we need to 
        # extend the attention mask to include all of the image token embeddings, but
        # exclude any of the padded text token embeddings.  In Huggingface notation,
        # 1 means keep and 0 means ignore
        image_attention_mask = torch.LongTensor([1] * image_emb.shape[1]).repeat(image_emb.shape[0], 1).to(image_emb.device)
        extended_attention_mask = torch.cat([text.attention_mask, image_attention_mask], 1)
        
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
            print(logits.view(-1, self.num_labels))
            print(labels.view(-1))
            print(loss)
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )


# In[ ]:


for batch in dataloader:
    outputs = model(**batch)


# In[ ]:


# Quick check if the forward inferencing works
model = MultimodalBertClassifier(num_labels=2)


# In[ ]:


# Print model info
num_params, size_all_mb = get_model_info(model)

print("Number of trainable params:", num_params)
print('Model size: {:.3f}MB'.format(size_all_mb))


# In[ ]:


# Setup the training arguments
output_dir = "./multimodal_hateful_memes"

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=32,
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


# In[ ]:


# Compute absolute learning rate
base_learning_rate = 1e-3
total_train_batch_size = (
    training_args.train_batch_size * training_args.gradient_accumulation_steps * training_args.world_size
)

training_args.learning_rate = base_learning_rate * total_train_batch_size / 256
print("Set learning rate to:", training_args.learning_rate)


# In[ ]:


# Create the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)


# In[ ]:


# Train
train_results = trainer.train()
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()


# In[ ]:


# Evaluate on the test dataset
metrics = trainer.evaluate(test_dataset)
trainer.log_metrics("test", metrics)


# In[ ]:



