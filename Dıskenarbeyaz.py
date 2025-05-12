import os
import cv2
import numpy as np

def process_images(input_folder):
    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            filepath = os.path.join(input_folder, filename)
            filepath = os.path.normpath(filepath)
            img = cv2.imread(filepath, cv2.IMREAD_COLOR)  # Renkli oku
            
            if img is None:
                print(f"Görüntü yüklenemedi: {filepath}")
                continue
            
            h, w, _ = img.shape  # Renkli olduğu için 3 kanal
            block_size = 16
            
            # Dış kenarlardaki blokları beyaz yap
            for i in range(0, h, block_size):
                for j in range(0, w, block_size):
                    if i == 0 or j == 0 or i >= h - block_size or j >= w - block_size:
                        img[i:i+block_size, j:j+block_size] = [255, 255, 255]
            
            save_path = os.path.join(input_folder, filename)
            cv2.imwrite(save_path, img)
            print(f"İşlenmiş resim kaydedildi: {save_path}")

# Örnek klasör yolu
input_folder = r"C:\Users\atade\Desktop\s11_3\desenler3"
process_images(input_folder)
