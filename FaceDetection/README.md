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
