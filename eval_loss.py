import torch
checkpoint = torch.load('./samples/SRGAN_x4-SRGAN_ImageNet/epoch_17.pth.tar')
print(checkpoint.keys())
