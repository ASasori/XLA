import os

output_dir = "./data/BSD100_images"
os.makedirs(output_dir, exist_ok=True)

for i, example in enumerate(ds["validation"]):
    hr = example["hr"]
    lr = example["lr"]
    
    hr.save(f"{output_dir}/hr_{i:03}.png")
    lr.save(f"{output_dir}/lr_{i:03}.png")

print(f"Saved {len(ds['validation'])} HR and LR images to {output_dir}")
