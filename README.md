# Custom YOLOv7 object detection -> Deepstream
Step by step guide to train SOTA Yolov7 models on custom data then accelerate and deploy on Nvidia Jetson through Deepstream. This repo will be absent of specific data or trained model for the time being to protect the sensitive nature of the model developed in the process of piecing together these components.

I put this together because as I started down my journey to train the YoloV7 model on a custom dataset I found that the repos that were available required heavy modification in order to work together. This is a culmination of learnings from across the official Yolov7 github repo found here (https://github.com/WongKinYiu/yolov7), the Yolo Deepstream repo found here (https://github.com/marcoslucianops/DeepStream-Yolo), and of course the Nvidia documentation for deepstream found here (https://docs.nvidia.com/metropolis/deepstream/dev-guide/).

## Gather Data
When gathering data there are 2 key considerations - are you using a dataset that already exists, or will you be gathering your own data for use in training?

If you will be using your own data you will have greater control over the content and complexity of the data which will lead to a more robust generalized model. When gathering data attempt to gather your dataset across as much of the image sensor as reasonable - i.e. take images of the objects in every corner of the image as well as the center. Don't be afraid to allow objects of interest to be obstructed and/or cut off at the edge of images.

Regardless of your choice of source dataset the YOLOv7 implenetation uses artificial distortions to increase model robustness.

Computer vision models cannot be built in a vaccuum without considering the devices which will be used to gather the data for training, and inference in production. This topic is an entire field of study independent of data science. Without turning this post into a discussion about professional imaging considerations let us complete this thought with the statement that computer vision models are sensitive to distortion, and all image gathering devices will have some amount of variation in the form of distortion, color, sharpness, clarity, etc. If you train a model using images of ideal quality with the subject in the center of the sensor your computer vision model will overfit and fail to be robust in inferencing. The computer vision community avoids overfitting by introducing artificial changes in the quality of images. Open source libraries used in the YOLOv7 implenetation use the OpenCV library to perterb the dataset prior to training. This allows your trained model to generalize more realiably - meaning when an object of interest enters the field of view during inference your model will have higher chance of successful detection even when the object is obstructed, warped, discolored, etc.

The distortions that your model will be trained under can be found in yolov7/data/hyp.scratch.custom.yaml hyperparameter tuning yaml file. These parameters include:

- hsv_h: 0.015  # image HSV-Hue augmentation (fraction)
- hsv_s: 0.7  # image HSV-Saturation augmentation (fraction)
- hsv_v: 0.4  # image HSV-Value augmentation (fraction)
- degrees: 0.0  # image rotation (+/- deg)
- translate: 0.2  # image translation (+/- fraction)
- scale: 0.9  # image scale (+/- gain)
- shear: 0.0  # image shear (+/- deg)
- perspective: 0.0  # image perspective (+/- fraction), range 0-0.001
- flipud: 0.0  # image flip up-down (probability)
- fliplr: 0.5  # image flip left-right (probability)
- mosaic: 1.0  # image mosaic (probability)
- mixup: 0.15  # image mixup (probability)



## Label Data using YOLO 1.1 format - CVAT
Install CVAT by navigating to https://cvat.org. I installed it as a containerized locally hosted server application and used it at localhost:8080. By default YOLOv7 is expecting your data to be labeled using rectangles and exported in yolo 1.1 format. More on the training/test data later. Be sure to label all objects of interest across the image - even when the object is obstructed and/or at edge of image.

## Train YoloV7 model
Copy provided loss.py file in ~/yolov7/utils. This file encorporates some bug fixes in handling the numpy indices.

From here you will want to create and activate a python virtual environment to avoid contaminating your OS's python installation in the event of package installation errors and dependency incompatability.

```
# apt install required packages
apt update
apt install -y zip htop screen libgl1-mesa-glx
```

```
# Clone yolov7 github repo
cd ~/
git clone https://github.com/WongKinYiu/yolov7.git
cd ~/yolov7
```

```
# Create and activate Python virtual environment:
python -m venv yolov7training
source yolov7training/bin/activate
```

```
# Install Python dependencies:
pip install -r requirements.txt
```

Copy provided loss.py file in ~/yolov7/utils. This file encorporates some bug fixes in handling the numpy indices.

The yolov7 implementation uses train and val folders that should be placed in the root directory of the project prior to training. Split your dataset into a typical 80/20 - train/val or some similar fraction. In both train and val create subfolders called images and labels and place the images and label files in their respective folders.

To train the model you will need to create a file under ~/yolov7/data called custom.yaml. You can copy and paste the file from this repo to that directory for use in training.

You will also need to download the respective yolov7 model that you wish to train and put it in the project root directory ~/yolov7.

```
# Training YoloV7 or V7x:
python train.py --workers 16 --device 0 --batch-size 16 --epoch 100 --data data/custom.yaml --img 640 640 --cfg cfg/training/yolov7x.yaml --weights 'yolov7x_training.pt' --name yolov7x-custom --hyp data/hyp.scratch.custom.yaml
```

```
# Training larger 1280x1280 models:
python train_aux.py --workers 8 --device 0 --batch-size 16 --epoch 100  --data data/custom.yaml --img 1280 --cfg cfg/training/yolov7-e6e.yaml --weights 'yolov7-e6e_training.pt' --name yolov7-e6e-custom --hyp data/hyp.scratch.custom.yaml
```

## Reparameterize model
Prior to deploying trained model for inference the model will need to be reparameterized. This process reduces size and complexity of the model without sacrificing accuracy or precision. 

Run respective reparameterization.ipynb section depending on which model was used in training. One important note is since these files are not all in the same directory you may need to modify the top of the notebook by adding these lines to the import statement section:
```
import sys
sys.path.insert(0, 'ABSOLUTE PATH TO /models')
sys.path.insert(1, 'ABSOLUTE PATH TO "yolov7 root"')
sys.path.insert(2, 'ABSOLUTE PATH TO /utils')
```

## Convert model for use in C++ using Deepstream
You will now have a trained yolov7 model ready for deploying on the edge for live inferencing. One available option for edge deployment is the Deepstream application built by Nvidia. A wonderful repo which can by used to build a C++ application for streaming exists here https://github.com/marcoslucianops/DeepStream-Yolo.

Copy your trained .pt model file to the root directory of the cloned Deepstream-yolo repo and execute .make file to build a C++ application which can be used for rapid live inference. C++ interacts with memory and hardware at a closer level in the OS than Python thereby increasing inference speed. Additional benefits of deploying your trained model in Deepstream is the built in metadata messaging broker which can communicate detected classes to Azure or other cloud platform of your choosing. See Nvidia Deepstream documentation for more detail on the message broker.
