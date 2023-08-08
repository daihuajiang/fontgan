# Handwriting to Calligraphy Style Transformation： A Generative Adversarial Network Approach
![](https://github.com/daihuajiang/fontgan/blob/main/img/architecture.png)

A modified model base on [zi2zi-pytorch](https://github.com/EuphoriaYan/zi2zi-pytorch). Adding a Cross-Attention block in bottleneck, which can improve the quality of generated results. A set of images is fed into the generator. This set of images includes the input character image and the skeleton input character image. You don't need to prepare the skeleton input character images. There is a function to process these steps in ./model/model.py. The functin will be executed in training and validation.

The generated results compare with other models:
![](https://github.com/daihuajiang/fontgan/blob/main/img/compare_with_other_model.png)

## How to Use
### Step Zero
Download tons of fonts as you please

### Requirement
* pytorch (>=1.0.0)
* numpy
* argparse
* scipy
* scikit-learn
* scikit-image
* matplotlib
* pillow
* imageio
* tqdm

### Preprocess
To avoid IO bottleneck, preprocessing is necessary to pickle your data into binary and persist in memory during training.

First run the below command to get the font images:
```python
python font2img.py --src_font=src.ttf
                   --dst_font=trg.otf
                   --charset=CN
                   --sample_count=1000
                   --sample_dir=dir
                   --label=0
                   --filter
                   --shuffle
```
Four default charsets are offered: CN, CN_T(traditional), JP, KR. You can also point it to a one line file, it will generate the images of the characters in it. Note, **filter** option is highly recommended, it will pre sample some characters and filter all the images that have the same hash, usually indicating that character is missing. **label** indicating index in the category embeddings that this font associated with, default to 0.

If you want validate the network with specific text, run the below command.
```python
python font2img.py --src_font=src.ttf
                   --dst_font=trg.otf
                   --charset=character.txt
                   --sample_count=1000
                   --sample_dir=dir
                   --label=0
```
**character.txt** should be a one line file.

### Package
After obtaining all images, run **package.py** to pickle the images and their corresponding labels into binary format:
```python
python package.py --dir=image_directories
                  --save_dir=binary_save_directory
                  --split_ratio=[0,1]
```
After running this, you will find two objects **train.obj** and **val.obj** under the **--save_dir** for training and validation, respectively.

### Experiment Layout
```python
experiment/
└── data
    ├── train.obj
    └── val.obj
```
Create a **experiment** directory under the root of the project, and a data directory within it to place the two binaries. Assuming a directory layout enforce better data isolation, especially if you have multiple experiments running.
### Train
To start training run the following command
```python
python train.py --experiment_dir=D:/dataset/zi2zi/202308test/ 
				--gpu_ids=cuda:0
				--batch_size=16
				--lr=0.001
				--epoch=30
				--sample_steps=1000
				--checkpoint_steps=1000
				--schedule=20
				--L1_penalty=100
				--Lconst_penalty=15 
```
**schedule** here means in between how many epochs, the learning rate will decay by half. The train command will create **sample,logs,checkpoint** directory under **experiment_dir** if non-existed, where you can check and manage the progress of your training.

During the training, you will find two or several checkpoint files **N_net_G.pth** and **N_net_D.pth** , in which N means steps, in the checkpoint directory.

**WARNING**, If your **--checkpoint_steps** is small, you will find tons of checkpoint files in you checkpoint path and your disk space will be filled with useless checkpoint file. You can delete useless checkpoint to save your disk space.

###　Infer
After training is done, run the below command to infer test data:
```python
python infer.py --experiment_dir experiment
                --batch_size 16
                --gpu_ids cuda:0 
                --resume {the saved model you select}
                --obj_pth obj_path
```
For example, if you want use the model 1000_net_G.pth and 1000_net_D.pth , which trained with 1000 steps, you should use --resume=1000.

## Acknowledgements
Code derived and rehashed from:

[pix2pix-tensorflow](https://github.com/yenchenlin/pix2pix-tensorflow) by [yenchenlin](https://github.com/yenchenlin)  
[Domain Transfer Network](https://github.com/yunjey/domain-transfer-network) by [yunjey](https://github.com/yunjey)  
[ac-gan](https://github.com/buriburisuri/ac-gan) by [buriburisuri](https://github.com/buriburisuri)  
[dc-gan](https://github.com/carpedm20/DCGAN-tensorflow) by [carpedm20](https://github.com/carpedm20)  
[origianl pix2pix torch code](https://github.com/phillipi/pix2pix) by [phillipi](https://github.com/phillipi)  
[zi2zi](https://github.com/kaonashi-tyc/zi2zi) by [kaonashi-tyc](https://github.com/kaonashi-tyc)  
[zi2zi-pytorch](https://github.com/xuan-li/zi2zi-pytorch) by [xuan-li](https://github.com/xuan-li)  
[Font2Font](https://github.com/yunchenlo/Font2Font) by [jasonlo0509](https://github.com/yunchenlo)  
[zi2zi-pytorch implement](https://github.com/EuphoriaYan/zi2zi-pytorch) by [EuphoriaYan](https://github.com/EuphoriaYan)

## License
Apache 2.0
