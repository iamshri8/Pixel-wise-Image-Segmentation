## Semantic Segmentation using U-Net architecture in Convolution Neural Networks. ##

The goal of semantic segmentation is to label every pixel in a given image to a corresponding class. Because the task involves pixel prediction, semantic segmentation is also called dense prediction.

DATASET: The dataset used for this project is called Cityscapes dataset. The dataset is downloaded from the official website. ​https://www.cityscapes-dataset.com/​. It focuses on semantic understanding of urban street scenes. The images are the result of extraction of thousands of frames from a moving vehicle during different months, covering spring, fall and summer in 50 different cities majorly in Germany and also its neighbouring countries.

The dataset consists of 30 different classes and are classified into various groups as shown below and out of 30 different classes only 6 classes namely car, road, person, sky, building, tree are considered and also background is considered as a class. So, totally the deep learning system in the project consists of 7 classes.


### Repository Overview ###

The files necessary for training are in train folder and the inferencing notebook is in the test folder.

1. train/

  - model.py -- script for model definition
  - pre_processing.py -- script for pre_processing the data during training
  - data_generator.py -- script for generating data for training in Keras
  - train.py -- script for training the deep learning model

2. test/

  - post_processing.py -- script has function definition for generating the segmentation map.
  - test.py -- script for model prediction on test images and outputting segmentation map.


### Training ###
Run the file train/train.py for training the model.

Command: python train/train.py
