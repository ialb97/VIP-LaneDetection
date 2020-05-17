# VIP-LaneDetection
LaneDetection for VIP smart cities course

If you would simply like to run the model simply clone this repo and use the checkpoint saved on github's LFS to run the model using your own images (replace the dataset object in test1.py with your own image data)



The workflow of the repo is as such Create TfRecords -> Load TfRecords -> train model -> test model


## Create TFRecords
The code to produce the TfRecords expects a csv file in the same format as train_all.csv, essentially this format is the path to the ground truth image in one column,the path to the binary image in another column, and the path to the segmentation image in the third column (this can be removed for the model currently implemented since it isn't utilized although it is included in the Tusimple dataset.

This csv file needs to be passed into the imagetoTfrecord.py file as well as a location for where the TFrecords will be stored. This python script will generate the tfrecords to be used with the rest of the code. 


## Load TFRecords & Train Model
The next step is to setup model.py with the parameters desired for training and have it load in the tfrecords generated using imagetoTfrecord.py. This will create the model and train it for the desired number of epochs while saving the model's weights every 25 epochs.


## Test Model
The python script test1.py is used to load a checkpoint into the model and run samples through it and get the results. This script is used to test and use the model.
