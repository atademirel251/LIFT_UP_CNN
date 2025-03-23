import os
import numpy as np
from PIL import Image

# Giriş klasörü ve çıktı klasörü
input_folder = r"C:\Users\atade\Desktop\10440_veri\input_Resim"  # PNG dosyalarının bulunduğu klasör
output_folder = r"C:\Users\atade\Desktop\output_Resim"  # Yeni çıktı dosyasının bulunduğu klasör

# Çıktı klasörünü oluştur (eğer yoksa)
os.makedirs(output_folder, exist_ok=True)

# Dosya listesini al ve sırala
file_list = sorted([f for f in os.listdir(input_folder) if f.endswith('.png')])
num_files = len(file_list)
print(f"Toplam {num_files} dosya bulundu.")

# Eğer hiç dosya yoksa işlemi durdur
if num_files == 0:
    raise ValueError("Belirtilen klasörde PNG dosyası bulunamadı.")

# Her bir PNG dosyasını oku ve dönüştür
for i, file_name in enumerate(file_list):
    file_path = os.path.join(input_folder, file_name)
    
    # PNG dosyasını aç
    with Image.open(file_path) as img:
        # Görüntüyü gri tonlamaya çevir
        img = img.convert("L") 
        
        # Görüntüyü 16x16 boyutuna küçült
        resized_image = img.resize((16, 16), Image.LANCZOS)
        
        # Görüntüyü tamamen siyah-beyaz yapmak için threshold uygula
        resized_array = np.array(resized_image)
        binary_image = np.where(resized_array < 128, 0, 255).astype(np.uint8)
        
        # Yeni çıktı klasörüne kaydet (resim ismi aynı)
        output_image_path = os.path.join(output_folder, file_name)  # Aynı ismi kullanarak yeni klasöre kaydet
        Image.fromarray(binary_image).save(output_image_path)  # PNG olarak kaydet
    
    # İlerlemeyi göster
    if (i + 1) % 100 == 0 or i == num_files - 1:
        print(f"{i + 1}/{num_files} dosya işlendi.")

print(f"Tüm resimler {output_folder} klasörüne kaydedildi.")
