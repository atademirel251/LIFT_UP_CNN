import cv2
import os

# PNG dosyalarının bulunduğu klasör yolu
folder_path = r"C:\Users\atade\Desktop\ek_dataset\resimler_28_net" # örnek: 'images'

# Klasördeki tüm dosyaları döngüyle gez
for filename in os.listdir(folder_path):
    if filename.endswith('.png'):
        file_path = os.path.join(folder_path, filename)

        # Görseli RGB (3 kanallı) olarak oku
        img_bgr = cv2.imread(file_path, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # img_rgb artık RGB formatında ve 3 kanallı
        print(f"{filename} - Boyut: {img_rgb.shape}")  # (yükseklik, genişlik, 3)
