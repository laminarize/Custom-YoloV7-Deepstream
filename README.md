#REPO IS WORK IN PROGRESS

# Custom YoloV7 object detection -> Deepstream
Step by step guide to train SOTA Yolov7 models on custom data then accelerate and deploy on Nvidia Jetson through Deepstream.

I put this together because as I started down my journey to train the YoloV7 model on a custom dataset I found that the repos that were available required heavy modification in order to work together. This is a culmination of learnings from across the official Yolov7 github repo found here (https://github.com/WongKinYiu/yolov7), the Yolo Deepstream repo found here (https://github.com/marcoslucianops/DeepStream-Yolo), and of course the Nvidia documentation for deepstream found here (https://docs.nvidia.com/metropolis/deepstream/dev-guide/).

## Gather Data
When gathering data there are 2 key considerations - are you using a dataset that already exists, or will you be gathering your own data for use in training?

If you will be using your own data you will have greater control over the content and complexity of the data which will lead to a more robust generalized model. When gathering data attempt to gather your dataset across as much of the image sensor as reasonable - i.e. take images of the objects in every corner of the image as well as the center. Don't be afraid to allow objects of interest to be obstructed and/or cut off at the edge of images.

Regardless of your choice of source dataset the YOLOv7 implenetation uses artificial distortions to increase model robustness.

Computer vision models cannot be built in a vaccuum without considering the devices which will be used to gather the data for training, and inference in production. This topic is an entire field of study independent of data science. Without turning this post into a discussion about professional imaging considerations let us complete this thought with the statement that computer vision models are sensitive to distortion, and all image gathering devices will have some amount of variation in the form of distortion, color, sharpness, clarity, etc. If you train a model using images of ideal quality with the subject in the center of the sensor your computer vision model will overfit and fail to be robust in inferencing. The computer vision community avoids overfitting by introducing artificial changes in the quality of images. Open source libraries used in the YOLOv7 implenetation use the OpenCV library to perterb the dataset prior to training. This allows your trained model to generalize more realiably - meaning when an object of interest enters the field of view during inference your model will have higher chance of successful detection even when the object is obstructed, warped, discolored, etc.

The distortions that your model will be trained under can be found in yolov7/data in the hyperparameter tuning yaml files. These include:

hsv_h: 0.015  # image HSV-Hue augmentation (fraction)

hsv_s: 0.7  # image HSV-Saturation augmentation (fraction)

hsv_v: 0.4  # image HSV-Value augmentation (fraction)

degrees: 0.0  # image rotation (+/- deg)

translate: 0.2  # image translation (+/- fraction)

scale: 0.9  # image scale (+/- gain)

shear: 0.0  # image shear (+/- deg)

perspective: 0.0  # image perspective (+/- fraction), range 0-0.001

flipud: 0.0  # image flip up-down (probability)

fliplr: 0.5  # image flip left-right (probability)

mosaic: 1.0  # image mosaic (probability)

mixup: 0.15  # image mixup (probability)


## Label Data using KITTI format - CVAT

## Train YoloV7 model

## Reparameterize model

## Convert model to C using Deepstream repo

## Launch live inference
