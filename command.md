Download weights
Download all available model weights.

# Download `SRGAN_x4-SRGAN_ImageNet.pth.tar` weights to `./results/pretrained_models`
$ bash ./scripts/download_weights.sh SRGAN_x4-SRGAN_ImageNet
# Download `SRResNet_x4-SRGAN_ImageNet.pth.tar` weights to `./results/pretrained_models`
$ bash ./scripts/download_weights.sh SRResNet_x4-SRGAN_ImageNet
# Download `DiscriminatorForVGG_x4-SRGAN_ImageNet.pth.tar` weights to `./results/pretrained_models`
$ bash ./scripts/download_weights.sh DiscriminatorForVGG_x4-SRGAN_ImageNet
Download datasets
These train images are randomly selected from the verification part of the ImageNet2012 classification dataset.

$ bash ./scripts/download_datasets.sh SRGAN_ImageNet
It is convenient to download some commonly used test data sets here.

$ bash ./scripts/download_datasets.sh Set5
How Test and Train
Both training and testing only need to modify yaml file.

Set5 is used as the test benchmark in the project, and you can modify it by yourself.

If you need to test the effect of the model, download the test dataset.

$ bash ./scripts/download_datasets.sh Set5
Test srgan_x4
$ python3 test.py --config_path ./configs/test/SRGAN_x4-SRGAN_ImageNet-Set5.yaml
Test srresnet_x4
$ python3 test.py --config_path ./configs/test/SRResNet_x4-SRGAN_ImageNet-Set5.yaml
Train srresnet_x4
First, the dataset image is split into several small images to reduce IO and keep the batch image size uniform.

$ python3 ./scripts/split_images.py
Then, run the following commands to train the model

$ python3 train_net.py --config_path ./configs/train/SRResNet_x4-SRGAN_ImageNet.yaml
Resume train srresnet_x4
Modify the ./configs/train/SRResNet_x4-SRGAN_ImageNet.yaml file.

line 33: RESUMED_G_MODEL change to ./samples/SRResNet_x4-SRGAN_ImageNet/g_epoch_xxx.pth.tar.
$ python3 train_net.py --config_path ./configs/train/SRResNet_x4-SRGAN_ImageNet.yaml
Train srgan_x4
$ python3 train_gan.py --config_path ./configs/train/SRGAN_x4-SRGAN_ImageNet.yaml
Resume train srgan_x4
Modify the ./configs/train/SRGAN_x4-SRGAN_ImageNet.yaml file.

line 38: PRETRAINED_G_MODEL change to ./results/SRResNet_x4-SRGAN_ImageNet/g_last.pth.tar.
line 40: RESUMED_G_MODEL change to ./samples/SRGAN_x4-SRGAN_ImageNet/g_epoch_xxx.pth.tar.
line 41: RESUMED_D_MODEL change to ./samples/SRGAN_x4-SRGAN_ImageNet/d_epoch_xxx.pth.tar.
$ python3 train_gan.py --config_path ./configs/train/SRGAN_x4-SRGAN_ImageNet.yaml
