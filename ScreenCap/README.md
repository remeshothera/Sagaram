# Sagaram
Pre-requisites

1. VLC player should be installed and added to PATH
2. Python with modules - sklearn , pyautogui,numpy , PIL 

svm.py
=================================
This script has two functions - classifyImage and executeMachine.
classifyImage will predict the image type and executeMachine will create a ml model and will pickle it.
Later we can use the pickled model for prediction.

VideoQualityAnalyser.py
==================================
This script will launch the reference video and take snapshots.

CreateDataset.py
==================================
This module will convert images to .csv data set file.This module will also downscale images with high resolution.