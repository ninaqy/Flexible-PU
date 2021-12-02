# Flexible-PU
This is the official implementation for "Deep Magnification-Flexible Upsampling over 3D Point Clouds" (https://ieeexplore.ieee.org/document/9555219). 

### Environment setting
The code is implemented with CUDA=10.0, tensorflow=1.14, python=2.7. Other settings should also be ok.

Other requested libraries: tqdm

### Compile tf_ops
One can refer to the setting of PUGeo-Net (https://github.com/ninaqy/PUGeo)

### Datasets and pretrained model
We provide the pretrained model and the training dataset that is capable of upsampling x4 to x16. Please download these files in the following link:
- training data in tfrecord form(train_data.tar)
- 39 testing models with 2048 points (test_data.tar) 
- pretrained x4 model (MAFU_model.tar) 

https://drive.google.com/drive/folders/1jLgGgfO8puQELgz9tbfwoLemGshjtSRl?usp=sharing

Please check the path for train and test data and modify the arguments "--train_data" and "--test_data" accordingly.


### Training
```
python main.py --phase train --r_train_list 4,8,12,16
```
Note that you can change the r_train_list to be the subset of [4,8,12,16], e.g. "--r_train_list 4" or "--r_train_list 8,12" etc.

### Inference
```
python main_upsample.py --phase test --pretrain --batch_size 1 --r_test_list 4 --log_dir model_pretrained_flexible
```
The upsampled xyz will be stored in "model_pretrained_flexible/test_x4".

Note that you can change the r_test_list to be the subset of [4,...,16], e.g. "--r_test_list 5" or "--r_test_list 7,8,16" etc.


We thank the authors of pointnet2, PU-Net, MPU and PU-GAN for their public code. 
