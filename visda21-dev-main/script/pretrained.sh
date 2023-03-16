python eval_pretrained_resnet.py --gpu_devices 5 6 --config ./configs/image_to_objectnet_imagenet_c_r_o.yaml --source_data /home/yzy/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train/ --target_data ./val_filelists/objectnet_c_r_o.txt

python eval_pretrained_resnet.py --gpu_devices 5 6 --config ./configs/image_to_imagenet_c_r_o.yaml --source_data /home/yzy/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train/ --target_data ./val_filelists/imagenet_c_r_o_filelist.txt

python eval_pretrained_resnet.py --gpu_devices 5 6 --config ./configs/image_to_objectnet.yaml --source_data /home/yzy/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train/ --target_data ./val_filelists/objectnet_filelist.txt