<div id="top" align="center">

# Only Positive Cases: 5-fold High-order Attention Interaction Model for Skin Segmentation Derived Classification
  
  Renkai Wu, Yinghao Liu, Pengchen Liang*, and Qing Chang* </br>
  
  [![arXiv](https://img.shields.io/badge/arXiv-2311.15625-b31b1b.svg)](https://arxiv.org/abs/2311.15625)

</div>

## News🚀
(2023.12.06) ***Prepare_ISIC2017.py file to include reminders. Please prepare the data according to the reminders, otherwise there may be an error in preparing the file***✅

(2023.12.01) ***The process and code for processing negative samples used to test the classification ability of the model is now online***🔥🔥

(2023.11.28) ***The arXiv paper version is publicly available***📃📃

(2023.11.28) ***You can download weight files of MHA-UNet here*** [Google Drive](https://drive.google.com/file/d/1LffUUhT1eiSVeLAOlLCg_cBbOjSaaows/view?usp=sharing) [Baidu Drive(btsd)](https://pan.baidu.com/s/1NjkumS8LaHJtTbOZxfqtkQ) . 🔥

(2023.11.26) ***The project code has been uploaded***🔥

(2023.11.25) ***The first edition of our paper has been uploaded to arXiv*** 📃

**0. Main Environments.**
- python 3.8
- pytorch 1.12.0

**1. Prepare the Dataset and Pretrained weights.**</br>

*A.Dataset* </br>
1- Download the ISIC 2017 train dataset from [this](https://challenge.isic-archive.com/data) link and extract both training dataset and ground truth folders inside the `/data/dataset_isic17/`. </br>
2- Run `Prepare_ISIC2017.py` for data preperation and dividing data to train,validation and test sets. </br>

**Notice:**</br>
For training and evaluating on ISIC 2018, pH2, NormalSkin and Kaggle95 follow the bellow steps: :</br>
1- Download the ISIC 2018 train dataset from [this](https://challenge.isic-archive.com/data) link and extract both training dataset and ground truth folders inside the `/data/dataset_isic18/`. </br> then Run ` Prepare_ISIC2018.py` for data preperation and dividing data to train,validation and test sets. </br>
2- Download the pH2 dataset from [this](https://www.dropbox.com/s/k88qukc20ljnbuo/PH2Dataset.rar) link and extract it then Run ` Prepare_PH2_test.py` for data preperation and dividing data to train,validation and test sets. </br>
3- Download the NormalSkin dataset from [this](https://universe.roboflow.com/janitha-prathapa/normalskin) link. </br>
4- Download the Kaggle95 dataset from [this](https://www.kaggle.com/datasets/ahdasdwdasd/our-normal-skin/data) link. </br>
5- The NormalSkin dataset and the Kaggle95 dataset were used as negative samples to test the model classification ability. For preparing these negative test samples, the data can be processed in the following way: </br>
#0 The negative dataset is without segmentation labels. It is possible to generate the same number of all-black labels as the original image in the following way: </br>
```
python generate_black.py
```
It should be noted that the number of generated images in the generate_black.py file needs to be modified. </br>

#1 When the same number of all-black labels are available, the following command is executed to generate test data: </br>
```
python Prepare_Neg_test.py
```
It should be noted that the number of images in the Prepare_Neg_test.py file needs to be modified. </br>

*B.Pretrained weights* </br>
The pretrained weights pth file can be obtained from  </br>
[Google Drive](https://drive.google.com/file/d/1LffUUhT1eiSVeLAOlLCg_cBbOjSaaows/view?usp=sharing). </br>
[Baidu Drive(btsd)](https://pan.baidu.com/s/1NjkumS8LaHJtTbOZxfqtkQ) </br>

**2. Train the MHA-UNet.** </br>
```
python train.py
```
- After trianing, you could obtain the outputs in './results/'

**3. Test the MHA-UNet.** </br>
First, in the test.py file, you should change the address of the checkpoint in 'resume_model' and fill in the location of the test data in 'data_path'.
```
python test.py
```
- After testing, you could obtain the outputs in './results/' </br>

**4. Get the MHA-UNet explainable results map and EICA calculations.** </br>
First, in the test_Explainable.py file, you should change the address of the checkpoint in 'resume_model' and fill in the location of the test data in 'data_path'.
```
python test_Explainable.py
```
- After testing, you could obtain the outputs in './results/'. EICA is calculated for each case. EICA threshold defaults to 225. The final display 'Detected as true(number):' is the number of all detected as positive. </br>

## Citation
If you find this repository helpful, please consider citing:
```
@article{wu2023only,
  title={Only Positive Cases: 5-fold High-order Attention Interaction Model for Skin Segmentation Derived Classification},
  author={Wu, Renkai and Liu, Yinghao and Liang, Pengchen and Chang, Qing},
  journal={arXiv preprint arXiv:2311.15625},
  year={2023}
}
```

## Acknowledgement </br>
This repo benefits from awesome works of [HorNet](https://github.com/raoyongming/HorNet), [MHorUNet](https://github.com/wurenkai/MHorUNet).

