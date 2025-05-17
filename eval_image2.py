import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from lpips import LPIPS
import torch
# from piq import niqe
import cv2

def calculate_metrics(img_sr, img_hr):
    """
    Tính toán các chỉ số PSNR, SSIM, LPIPS và NIQE giữa ảnh SR và ảnh HR.

    Args:
        img_sr (np.ndarray): Ảnh super-resolution (H, W, C) hoặc (N, H, W, C).
        img_hr (np.ndarray): Ảnh gốc độ phân giải cao (H, W, C) hoặc (N, H, W, C).

    Returns:
        dict: Một dictionary chứa các giá trị PSNR, SSIM, LPIPS và NIQE.
    """
    # Đảm bảo ảnh có cùng kích thước và kiểu dữ liệu
    if img_sr.shape != img_hr.shape:
        raise ValueError("Kích thước của ảnh SR và HR không khớp.")

    # Chuẩn hóa ảnh về dải [0, 1] nếu cần
    if img_sr.max() > 1.0:
        img_sr = img_sr / 255.0
    if img_hr.max() > 1.0:
        img_hr = img_hr / 255.0

    # Tính PSNR
    psnr_val = peak_signal_noise_ratio(img_hr, img_sr, data_range=1.0)

    # Tính SSIM
    ssim_val = structural_similarity(img_hr, img_sr, channel_axis=-1, data_range=1.0, win_size=11)

    # Tính LPIPS
    lpips_scorer = LPIPS(net='vgg').to('cuda' if torch.cuda.is_available() else 'cpu')
    img_sr_tensor = torch.from_numpy(img_sr).permute(2, 0, 1).unsqueeze(0).float().to(lpips_scorer.device)
    img_hr_tensor = torch.from_numpy(img_hr).permute(2, 0, 1).unsqueeze(0).float().to(lpips_scorer.device)
    lpips_val = lpips_scorer(img_sr_tensor, img_hr_tensor).item()

    # Tính NIQE (chỉ áp dụng cho ảnh HR)
    # Đảm bảo ảnh HR là ảnh đơn lẻ (không phải batch) và có 3 kênh màu
    # if img_hr.ndim == 3 and img_hr.shape[2] == 3:
    #     img_hr_tensor_niqe = torch.from_numpy(img_hr).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    #     niqe_val = niqe(img_hr_tensor_niqe).item()
    # else:
    #     niqe_val = np.nan  # Hoặc một giá trị phù hợp khác

    return {
        "PSNR": psnr_val,
        "SSIM": ssim_val,
        "LPIPS": lpips_val,
        #"NIQE": niqe_val
    }

# Giả sử bạn đã load ảnh SR và HR vào các biến img_sr và img_hr (dưới dạng mảng NumPy)
# Ví dụ:
img_sr = cv2.imread('./data/SRGAN_ImageNet/ILSVRC2012_val_00000045.JPEG')
img_hr = cv2.imread('./figure/srganx4_dog_lastest.jpg')

# Tính toán các chỉ số
results = calculate_metrics(img_sr, img_hr)

# In kết quả
print(f"PSNR: {results['PSNR']:.2f} dB")
print(f"SSIM: {results['SSIM']:.4f}")
print(f"LPIPS: {results['LPIPS']:.4f}")
#print(f"NIQE: {results['NIQE']:.4f}")