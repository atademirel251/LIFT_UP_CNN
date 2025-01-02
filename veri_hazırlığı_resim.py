import os
from PIL import Image
import numpy as np

def convert_images_to_grayscale(input_folder, output_folder, target_size=(32, 32)):
    """
    Belirtilen klasördeki resimleri 32x32 boyutunda ve gri tonlamalı hale dönüştürür.
    
    Args:
        input_folder (str): Orijinal resimlerin bulunduğu klasör.
        output_folder (str): Dönüştürülen resimlerin kaydedileceği klasör.
        target_size (tuple): Hedef boyut (genişlik, yükseklik).
        
    Returns:
        np.ndarray: Dönüştürülen resimlerin numpy dizisi.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    image_arrays = []
    
    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            filepath = os.path.join(input_folder, filename)
            
            # Resmi yükle ve gri tonlamalıya dönüştür
            image = Image.open(filepath).convert('L')
            
            # Resmi yeniden boyutlandır
            resized_image = image.resize(target_size)
            
            # Resmi numpy array'e çevir ve 0-1 aralığına normalizasyon yap
            image_array = np.array(resized_image) / 255.0
            
            # Diziye ekle
            image_arrays.append(image_array)
            
            # Yeni resmi kaydet
            save_path = os.path.join(output_folder, filename)
            resized_image.save(save_path)
    
    # Tüm resimleri 4D numpy array'e dönüştür (ör. [adet, 32, 32, 1])
    image_arrays = np.expand_dims(np.array(image_arrays), axis=-1)
    return image_arrays

# Kullanım:
input_folder = "input_images"  # Orijinal resimlerin bulunduğu klasör
output_folder = "output_images"  # Dönüştürülen resimlerin kaydedileceği klasör

converted_images = convert_images_to_grayscale(input_folder, output_folder)
print(f"Dönüştürülen resimlerin boyutu: {converted_images.shape}")
