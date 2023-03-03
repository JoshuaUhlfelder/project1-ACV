# Using Pre-Trained Deep Learning Models to Identify Underlying Diseases from Skin Lesion Image Data
ACV-project1
Joshua Uhlfelder
# 

README.md

The goal of this project is to develop a model that identifies the underlying disease (melanoma, dermatosisroma, carcinoma, etc.) from images of skin lesions. Three different types of pre-trained neural network models were fine-tuned and applied to skin lesion image data from the HAM10000 dataset (https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

In this repo, there are four types of files: 
1. Model training files (model1.py, model2.py,...) - These have editable parameter to train models on the trianing data and evaluate on the validation data.
2. Testing files (m1test.py, m2test.py) - These run test data on the models derived from the model training files to get metrics.
3. Toe files (m1toe.py, m2toe.py) - These can be used to run data through previously derived models to see outputs. The name comes from me inspecting a mole on my toe to classify it. The datafile toe_metadata should contain image information. The underlying directory should have a folder with the actual images you want to run.
4. Data files (HAM10000_metadata.csv, toe_metadata.csv) - These contain data regarding the data files

Note: Data files should be added in the underlying directory, or model/testing/toe files can be modified to take in data from any directory. If using the HAM10000 dataset, create a folder with all and only image data named 'HAM10000_images'. Put this in the underlying directoy. Add the HAM10000_metadata file in the same directory as the model you are running.

# Model training files
model1.py - trains a ResNet50 pretrained from ImageNet to classify skin lesions
model2.py - trains a ViT pretrained from ImageNet to classify skin lesions
model3.py - trains a BERT with a pretrained ResNet50 image encoder and a custom information encoder with demographic data about the skin lesion
model4.py - trains a binary classification BERT with a pretrained ResNet50 image encoder and a custom information encoder
model5.py - trains a BERT with a pretrained ResNet50 image encoder and a BERT text encoder with demographic data about the skin lesion

model3.py does not have an accompanying test or toe file, as evaluation is completed after training in the script
Models 4-5 showed not difference in performance to model3, so they also lack evaluation scripts.

# Testing files
1. m1test.py - outputs metrics after running test data on model1 created from model1.py
2. m2test.py - outputs metrics after running test data on model2 created from model2.py


# Toe files
1. m1toe.py - classifies data on model1
2. m2toe.py - classifies data on model2


# Data files
HAM10000_metadata.csv and toe_metadata.csv should have the same format. HAM10000_metadata can be downloaded with the HAM10000 dataset. Both should be in the pwd.

