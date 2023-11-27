<div id="top" align="center">

# MHA-UNet
**Only Positive Cases: 5-fold High-order Attention Interaction Model for Skin Segmentation Derived Classification**
  
  Renkai Wu, Yinghao Liu, Pengchen Liang*, and Qing Chang*

</div>

**0. Main Environments**
- python 3.8
- pytorch 1.8.0
- torchvision 0.9.0

**1. Prepare the dataset.**

1- Download the ISIC 2017 train dataset from [this](https://challenge.isic-archive.com/data) link and extract both training dataset and ground truth folders inside the `/data/dataset_isic17/`. </br>
2- Run `Prepare_ISIC2017.py` for data preperation and dividing data to train,validation and test sets. </br>

**Notice:**
For training and evaluating on ISIC 2018, pH2, NormalSkin and Kaggle95 follow the bellow steps: :</br>
1- Download the ISIC 2018 train dataset from [this](https://challenge.isic-archive.com/data) link and extract both training dataset and ground truth folders inside the `/data/dataset_isic18/`. </br> then Run ` Prepare_ISIC2018.py` for data preperation and dividing data to train,validation and test sets. </br>
2- Download the pH2 dataset from [this](https://www.dropbox.com/s/k88qukc20ljnbuo/PH2Dataset.rar) link and extract it then Run ` Prepare_PH2_test.py` for data preperation and dividing data to train,validation and test sets. </br>
3- Download the NormalSkin dataset from [this](https://universe.roboflow.com/janitha-prathapa/normalskin) link.
4- Download the Kaggle95 dataset from [this](https://www.kaggle.com/datasets/ahdasdwdasd/our-normal-skin/data) link.


**2. Train the MHA-UNet.**
```
python train.py
```
- After trianing, you could obtain the outputs in './results/'

**3. Test the MHA-UNet.**
First, in the test.py file, you should change the address of the checkpoint in 'resume_model' and fill in the location of the test data in 'data_path'.
```
python test.py
```
- After testing, you could obtain the outputs in './results/'

**4. Get the MHA-UNet explainable results map and EICA calculations.**
First, in the test_Explainable.py file, you should change the address of the checkpoint in 'resume_model' and fill in the location of the test data in 'data_path'.
```
python test_Explainable.py
```
- After testing, you could obtain the outputs in './results/'. EICA is calculated for each case. EICA threshold defaults to 225. The final display 'Detected as true(number):' is the number of all detected as positive.
