import os
import numpy as np
from PIL import Image

# Giriş klasörü ve çıktı dosyası
input_folder = r"C:\Users\atade\Desktop\10440_veri\input_Resim"  # PNG dosyalarının bulunduğu klasör
output_file = r"C:\Users\atade\Desktop\ahmet.npy"  # Çıktı dosyası ('.npy' eklendi)

# Dosya listesini al ve sırala
file_list = sorted([f for f in os.listdir(input_folder) if f.endswith('.png')])
num_files = len(file_list)
print(f"Toplam {num_files} dosya bulundu.")

# Eğer hiç dosya yoksa işlemi durdur
if num_files == 0:
    raise ValueError("Belirtilen klasörde PNG dosyası bulunamadı.")

# Tüm dönüştürülmüş görüntüleri saklamak için boş bir liste
all_images = []

# Her bir PNG dosyasını oku ve dönüştür
for i, file_name in enumerate(file_list):
    file_path = os.path.join(input_folder, file_name)
    
    # PNG dosyasını aç
    with Image.open(file_path) as img:
        # Görüntüyü gri tonlamaya çevir
        img = img.convert("L") 
        
        # Görüntüyü 16x16 boyutuna küçült
        resized_image = img.resize((16, 16), Image.LANCZOS)
        
        # Numpy dizisine çevir
        resized_array = np.array(resized_image)
        
        # Görüntüyü tamamen siyah-beyaz yapmak için threshold uygula
        binary_image = np.where(resized_array < 128, 0, 255).astype(np.uint8)
        
        # Dönüştürülmüş görüntüyü listeye ekle
        all_images.append(binary_image)
    
    # İlerlemeyi göster
    if (i + 1) % 100 == 0 or i == num_files - 1:
        print(f"{i + 1}/{num_files} dosya işlendi.")

# Tüm görüntüleri numpy array'e dönüştür
all_images = np.array(all_images)

# Çıktıyı bir dosyaya kaydet
np.save(output_file, all_images)
print(f"Tüm veriler {output_file} dosyasına kaydedildi.")

# Kaydedilen dosyayı yükleyip kontrol et
loaded_data = np.load(output_file)
print(f"Yüklenen veri boyutu: {loaded_data.shape}")
