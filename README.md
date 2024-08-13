# Facial Emotion Recognition Using EAC
## Objective
This repository is dedicated to a Facial Emotion Recognition project that leverages the cutting-edge EAC technique, utilizing two primary backbones: ResNet50 and MobileNet_V2. The core objective of this project is to achieve emotion recognition through live video with exceptional accuracy—exceeding 90%. This approach not only demonstrates superior precision but also outperforms existing state-of-the-art methods, showcasing a significant advancement over conventional models.


## Approach
**Erasing Attention Consistency (EAC)** explore dealing with noisy labels from a new feature-learning perspective. FER models remember noisy samples by focusing on a part of the features that can be considered related to the noisy labels instead of learning from the whole features that lead to the latent truth. We use this method to suppress the noisy samples during the training process automatically. EAC significantly outperforms state-of-the-art noisy label FER methods and generalizes well to other tasks.


## Dataset

We are using public dataset **RAF-DB** for training and evaluating our model. 

## Train the EAC model

We have trained our model with two backbones: **ResNet50** and **MobileNet_V2**

We have set up all requirement in enviroment on **Kaggle**. All the log and results during the training and testing process have been saved on [Wandb.ai](https://wandb.ai/hung123ka5/Facial%20Emotion%20Recognition?nw=nwuserhung123ka5)

You can follow these link to get the source code:

[Training_EAC_With_Backbone_ResNet50](https://www.kaggle.com/code/hunghoang31/eac-resnet-ipynb)

[Training_EAC_With_Backbone_MobileNet_V2](https://www.kaggle.com/code/hunghoang31/eac-mobilenet-v2-80-ipynb/notebook)

## Accuracy

Traing EAC on RAF-DB clean train set (ResNet-50 backbone) should achieve over 90.35\% accuracy on RAF-DB test set.

## Set up Enviroment for UI

You need to install libraries in files "ui.py", "model.py" and "model2.py" by command examples:
```
pip install torch
```

Download the pretrained [ResNet-50 model](https://drive.google.com/file/d/1yQRdhSnlocOsZA4uT_8VO0-ZeLXF4gKd/view?usp=sharing) and then put it under the model directory. 

Download the final pretrained weight [ResNet50_final_weight](https://drive.google.com/file/d/1bvcidRXkAxxEYXmcaD8LUQues81cEzIy/view?usp=sharing) then put it under the model directory 

```key
- /UI demo 
      model/
          resnet50_ft_weight.pkl (the file you have downloaded)
          resnet50_final.pth     (the file you have downloaded)
          mobilenet_v2_final.pth
        
```


## Project Structure
```
├── ...
├── Training
│  ├── ...                 # You can access the Kaggle link above  
├── UI demo
│  ├── data                # Images to recognize emotion
│  ├── model               # Model's pretrained weight files
│  ├── model.py            # Build Model with ResNet50 as backbone
│  ├── model2.py           # Build Model with MobileNet_V2 as backbone
│  ├── ui.py               # UI 
|  ├── ...
└── ...

```