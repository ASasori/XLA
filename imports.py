import time
import torch
from model import SRResNet
import os

print("---\n\nLoading local SR models...\n\n---")
start = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models = {
    "SRGAN_x4_Div2K": "results/SRGAN_x4-SRGAN_Div2K/g_best.pth.tar",
    "SRGAN_x4_ImageNet": "results/SRGAN_x4-SRGAN_ImageNet/g_best.pth.tar",
    "SRResNet_x4_Div2K": "results/SRResNet_x4-SRGAN_Div2K/g_best.pth.tar",
    "SRResNet_x4_ImageNet": "results/SRResNet_x4-SRGAN_ImageNet/g_best.pth.tar",
}

# def load_model(path, model_type="srresnet"):
#     checkpoint = torch.load(path, map_location=device)
#     state_dict = checkpoint["state_dict"]

#     # Loại bỏ prefix "_orig_mod." nếu tồn tại
#     new_state_dict = {}
#     for k, v in state_dict.items():
#         new_k = k.replace("_orig_mod.", "")
#         new_state_dict[new_k] = v

#     if model_type.lower() == "srresnet":
#         model = SRResNet(upscale=4)
#     else:
#         model = SRResNet(upscale=4)  # Có thể đổi sau nếu bạn có mô hình SRGAN

#     model.load_state_dict(new_state_dict)
#     model = model.to(device).eval()
#     return model

# Load toàn bộ mô hình
loaded_models = models 
# for name, path in models.items():
#     model_type = "SRResNet" if "SRResNet" in name else "SRGAN"
#     print(f"Loading {name} ...")
#     loaded_models[name] = load_model(path, model_type=model_type)

print(f"\n---\nLoaded all models in {round(time.time() - start, 2)} seconds\n---")
