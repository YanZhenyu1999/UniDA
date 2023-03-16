#!/bin/bash

python train_ovanet.py --gpu_devices 6  --config ./configs/image_to_objectnet.yaml --source_data /home/yzy/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train/ --target_data ./val_filelists/objectnet_filelist.txt --multi 0.1

python train_ovanet.py --gpu_devices 6  --config ./configs/image_to_objectnet.yaml --source_data /home/yzy/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train/ --target_data ./val_filelists/objectnet_filelist.txt --multi 0.05

python train_ovanet.py --gpu_devices 6  --config ./configs/image_to_objectnet.yaml --source_data /home/yzy/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train/ --target_data ./val_filelists/objectnet_filelist.txt --multi 0.02

python train_ovanet.py --gpu_devices 6  --config ./configs/image_to_objectnet.yaml --source_data /home/yzy/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train/ --target_data ./val_filelists/objectnet_filelist.txt --multi 0.01