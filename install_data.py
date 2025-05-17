from datasets import load_dataset
import os
from PIL import Image
import numpy as np

# Load dataset từ HuggingFace và set định dạng đầu ra là numpy
ds = load_dataset("eugenesiow/BSD100", "bicubic_x4", cache_dir="./data")
ds.set_format(type="numpy", columns=["hr", "lr"])

output_dir = "./data/BSD100_images"
os.makedirs(output_dir, exist_ok=True)

for i, example in enumerate(ds["validation"]):
    hr_img = Image.fromarray(example["hr"])
    lr_img = Image.fromarray(example["lr"])

    hr_img.save(f"{output_dir}/hr_{i:03}.png")
    lr_img.save(f"{output_dir}/lr_{i:03}.png")

print(f"Saved {len(ds['validation'])} HR and LR images to {output_dir}")
