# Custom YoloV7 object detection -> Deepstream
Step by step guide to train SOTA Yolov7 models on custom data then accelerate and deploy on Nvidia Jetson through Deepstream.

I put this together because as I started down my journey to train the YoloV7 model on a custom dataset I found that the repos that were available required heavy modification in order to work together. This is a culmination of learnings from across the official Yolov7 github repo found here (https://github.com/WongKinYiu/yolov7), the Yolo Deepstream repo found here (https://github.com/marcoslucianops/DeepStream-Yolo), and of course the Nvidia documentation for deepstream found here (https://docs.nvidia.com/metropolis/deepstream/dev-guide/).

## Gather Data

## Label Data using KITTI format - CVAT

## Train YoloV7 model

## Reparameterize model

## Convert model to C using Deepstream repo

## Launch live inference
