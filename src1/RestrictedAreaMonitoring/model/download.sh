#!/bin/bash
# Stock Ultralytics YOLO11s, COCO-pretrained. The same detector CustomerHeatmaps,
# QueueLengthMonitoring and VehicleDetection use: verified identical weights
# (initializer sha1 2ba6a9c05758, 80 classes, 321 nodes) - only the embedded
# export metadata differed between copies, which is why their file hashes did.
#
# Fetched as a ready ONNX rather than exported here. The previous version ran
# `pip install ultralytics` and exported at download time, which pulls ~800 MB of
# torch and measured 252 s in the DeepStream container. PipeGen kills a download
# script at 300 s and swallows the error, so on any slower link the model simply
# never appeared and the run stopped to ask for a model path.
cd "$(dirname "$0")" || exit 1
gdown --fuzzy https://drive.google.com/file/d/1OQIrwowlFJ65QMT12yL1OmDMLM1NKZm-/view
