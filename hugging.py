import time
start = time.time()
from flask import Flask, render_template, request, send_file, send_from_directory
import torch
from PIL import Image
import os
from torchvision import transforms
from imports import loaded_models  # Sử dụng models đã load sẵn
import subprocess
import uuid

print(f"---\n\nTime to load libraries: {round(time.time() - start, 2)} seconds\n\n---")

os.makedirs("outputs", exist_ok=True)

app = Flask(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"---\n\nDevice: {device}\n\n---")

# Lấy danh sách model đã load sẵn (chứa đường dẫn)
models = loaded_models
current_model_name = list(models.keys())[0]  # mặc định
current_model_WEIGHT_PATH = models[current_model_name]


def get_model_architecture(model_name):
    if "SRResNet" in model_name:
        return "SRResNet"
    elif "SRResNet" in model_name:
        return "SRResNet"
    return None

@app.route('/')
def index():
    return render_template('index.html', available_models=models.keys())


@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return render_template('index.html', error='No image file')

    file = request.files['image']
    if file.filename == '':
        return render_template('index.html', error='No selected image file')

    input_path = os.path.join("outputs", "input.png")
    output_path = os.path.join("outputs", "upscaled.png")
    print(f"Hugging.py - Output path (before subprocess): {os.path.abspath(output_path)}")


    # Xoá file cũ nếu có
    if os.path.exists(output_path):
        os.remove(output_path)

    file.save(input_path)
    arch = "SRResNet"
    # # Lấy kiến trúc model từ tên
    # model_architecture = get_model_architecture(current_model_name)
    # if not model_architecture:
    #     return render_template('index.html', error=f"Không thể xác định kiến trúc cho model: {current_model_name}")

    # Chạy inference.py qua subprocess
    subprocess.run([
        "python3", "inference.py",
        "--inputs", input_path,
        "--output", output_path,
        "--model_arch_name", arch,  # Truyền kiến trúc model
        "--model_weights_path", current_model_WEIGHT_PATH,
        "--device", str(device)
    ])

    if not os.path.exists(output_path):
        return render_template('index.html', error='Failed to generate image')

    print("---\n\nUpscaled image saved as upscaled.png\n\n---")
    return render_template('index.html', result="upscaled.png")


@app.route('/select_model', methods=['POST'])
def select_model():
    global current_model_name
    global current_model_WEIGHT_PATH
    selected_model = request.form.get('model_select')
    if selected_model in models:
        current_model_name = selected_model
        current_model_WEIGHT_PATH = models[current_model_name]
    return render_template('index.html', available_models=models.keys())


@app.route('/outputs/<filename>')
def uploaded_file(filename):
    return send_from_directory("outputs", filename)


if __name__ == '__main__':
    import torch
    app.run(debug=True)