# test_PhD

Code by **Nguyen Tuan Nam**

## 1. Introduction
This is source code reimplement a paper Cutting the Error by Half: Investigation of Very Deep CNN and Advanced Training
Strategies for Document Image Classification (https://arxiv.org/abs/1704.03557). This system is used to recognize type of document on RVL_CDIP dataset and Tobacco_3482 dataset and written by Python 3


## 2. Installation

This software depends on NumPy, Keras, Tensorflow, matplotlib, opencv-python. You must have them installed before using.
The simple way to install them is using pip: 
```sh
	# sudo pip3 install -r requirements.txt
```
We also provide **Dockerfile** to deploy environtment 

## 3. Usage

### 3.1. Data
Downloading RVL_CDIP dataset (https://www.cs.cmu.edu/~aharley/rvl-cdip/) and Tobacco dataset(https://www.kaggle.com/patrickaudriaz/tobacco3482jpg). And extract all downloaded files(rvl-cdip.tar.gz, labels_only.tar.gz, tobacco3482jpg.zip) in same folder of source.

After that, we run create_dataset.py by a following command: 
```sh
	# python3 create_dataset.py
```
This command will move all image with same label to same folder in rvl-cdip dataset and remove all image of rvl-cdip training dataset which is contained in tobaco3482.

### 3.2.Training

### 3.2.1. Train and test RVC_CDIP dataset
