import cv2
lr_path = './figure/0031.png'
def load_image(image_path, target_size=(96, 96)):
    img = cv2.imread(image_path)  # Đọc ảnh
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Chuyển đổi từ BGR sang RGB
    img = cv2.resize(img, target_size)  # Thay đổi kích thước ảnh
    img = img / 255.0  # Chuẩn hóa ảnh về [0, 1]
    return img

# Tải ảnh HR và LR
lr_image = load_image(lr_path, target_size=(96, 96))
#save image 
cv2.imwrite('./figure/lr_0031.png', (lr_image * 255).astype('uint8'))
