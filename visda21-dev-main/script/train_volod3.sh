python train_ovanet.py --gpu_devices 2 --network volod3  --source_data /home/yzy/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train/ --target_data ./val_filelists/objectnet_c_r_o.txt --multi 0.01

python train_ovanet.py --gpu_devices 6 --network volod3  --config ./configs/image_to_imagenet_c_r_o.yaml --source_data /home/yzy/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train/ --target_data ./val_filelists/imagenet_c_r_o_filelist.txt --multi 0.01

python train_ovanet.py --gpu_devices 6 --network volod3  --config ./configs/image_to_objectnet.yaml --source_data /home/yzy/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train/ --target_data ./val_filelists/objectnet_filelist.txt --multi 0.01