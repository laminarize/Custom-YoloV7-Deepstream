# Custom YoloV7 object detection -> Deepstream
Step by step guide to train SOTA Yolov7 models on custom data then accelerate and deploy on Nvidia Jetson through Deepstream.

To complete this guide you will need: 

  1. Machine with CUDA enabled GPU. YoloV7 standard with 8 workers and batch size of 16 used ~20GB of VRAM in training.
  2. Nvidia Jetson device with Deepstream 6.1/6.1.1 installed

## Gather Data

## Label Data using KITTI format - CVAT

## Train YoloV7 model

## Reparameterize model

## Convert model to C using Deepstream repo

## Launch live inference
