# Credits

This project uses DETR (DEtection TRansformer) developed by Facebook Research. We acknowledge and are grateful to Facebook Research for their contributions to the field of object detection and machine learning. For more information about DETR, visit the [official GitHub repository](https://github.com/facebookresearch/detr).

# The steps to run the face detection model:

## Step 1:
Download and Unzip WIDER Face Dataset  
In a new code cell, run the following to download WIDER Face dataset from Google Drive links:

```
!pip install --upgrade --no-cache-dir gdown
!gdown --id 1nKot1S6ARpB6wnyx3R1tefduUiUIjO9p
!gdown --id 1rkUK6OG7js5gYt1bhCUanPNqmuxwkVZX
```

You will have downloaded 2 zip files, WIDER_val.zip and WIDER_train.zip  
Unzip the files to ./data:

```
!unzip -q WIDER_train.zip -d ./data
!unzip -q WIDER_val.zip -d ./data
```

## Step 2:
Unzip the wider_face_split.zip, this is the annotation files for dataset

## Step 3:
Clone the DETR Repository on Colab  
In the colab notebook, run the following commands:  
```
!git clone https://github.com/facebookresearch/detr.git
import os
os.chdir('detr')
!git checkout 8a144f83a287f4d3fece4acdf073f387c5af387d
```
This will clone the DETR repository at a specific commit in the colab’s notebook environment.  

## Step 4:
Training the model  
Training the model is quite simple. DETR already has the training code written for you in main.py. Run this command line:  
```
!python main.py --coco_path './data' --epochs 2 --batch_size 3 --resume 'https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth' --output_dir './ckpts'
```
This will retrain the model starting from the pretrained weights (--resume).  
The new weights will be saved in ./ckpts  
Each epoch of training takes about 3 hours on Colab. You can play with different epochs and batch_size.
After this, you can start training from your checkpoints.
