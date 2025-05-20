import cv2
import torch
import lpips
##from piq import niqe  # Import the niqe function directly
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage import img_as_float32
# from piq.quality_metrics import niqe
# Đọc ảnh ground-truth và ảnh SR
#gt = cv2.imread("./data/SRGAN_ImageNet/ILSVRC2012_val_00000045.JPEG")
# gt = cv2.imread("./figure/ILSVRC2012_val_00000212.JPEG")
gt = cv2.imread("./data/Set5/X4/GT/head.png")
#sr = cv2.imread("./figure/srganx4_dog_lastest.jpg")
# sr = cv2.imread("./figure/sr_ILSVRC2012_val_00000212.JPEG")
sr = cv2.imread("./outputs/upscaled.png")
# Resize ảnh ground-truth nếu kích thước không khớp
gt = cv2.resize(gt, (sr.shape[1], sr.shape[0]))

# Chuyển sang RGB
gt_rgb = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
sr_rgb = cv2.cvtColor(sr, cv2.COLOR_BGR2RGB)

# Tính PSNR & SSIM
psnr_val = psnr(gt_rgb, sr_rgb, data_range=255)
ssim_val = ssim(gt_rgb, sr_rgb, channel_axis=-1, data_range=255, win_size=3)

# Tính LPIPS (chuyển ảnh sang tensor [-1,1] và [C,H,W])
loss_fn = lpips.LPIPS(net='alex')  # hoặc 'vgg'
gt_tensor = torch.tensor(gt_rgb).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
sr_tensor = torch.tensor(sr_rgb).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
lpips_val = loss_fn(gt_tensor, sr_tensor).item()

# Tính NIQE bằng PIQ (trên ảnh SR grayscale float32)
sr_gray = cv2.cvtColor(sr_rgb, cv2.COLOR_RGB2GRAY)
sr_gray_float = img_as_float32(sr_gray)
sr_gray_tensor = torch.tensor(sr_gray_float).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
# niqe_val = niqe(sr_gray_tensor, data_range=1.0).item()

# In kết quả
print(f"PSNR: {psnr_val:.2f} dB")
print(f"SSIM: {ssim_val:.4f}")
print(f"LPIPS: {lpips_val:.4f}")
# print(f"NIQE: {niqe_val:.4f}")
