import os
import numpy as np
import cv2

def clean_8x8_blocks(image, threshold=0.5):
    """Post-processing to clean up 8x8 blocks for 16x16 block layout"""
    h, w = image.shape[:2]
    cleaned = np.zeros_like(image)

    for y in range(0, h, 8):
        for x in range(0, w, 8):
            block = image[y:y+8, x:x+8]
            avg = np.mean(block)
            value = 255 if avg > threshold * 255 else 0  # 255: beyaz, 0: siyah
            cleaned[y:y+8, x:x+8] = value

    return cleaned

# Klasör yolu
giris_klasoru = r"C:\Users\atade\Desktop\13579_veri\resim128_opencv_rgb"
cikis_klasoru = r"C:\Users\atade\Desktop\13579_veri\resim128_NET"

# Çıkış klasörü yoksa oluştur
os.makedirs(cikis_klasoru, exist_ok=True)

# Klasördeki her dosya için dön
for dosya_adi in os.listdir(giris_klasoru):
    if dosya_adi.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        dosya_yolu = os.path.join(giris_klasoru, dosya_adi)
        
        # Görüntüyü oku (siyah-beyaz olarak)
        img = cv2.imread(dosya_yolu, cv2.IMREAD_GRAYSCALE)
        
        if img.shape != (128, 128):
            print(f"Uyarı: {dosya_adi} boyutu {img.shape}, 128x128 değil, atlanıyor.")
            continue

        # Blok temizleme uygula
        temizlenmis = clean_8x8_blocks(img)

        # Kaydet
        kayit_yolu = os.path.join(cikis_klasoru, dosya_adi)
        cv2.imwrite(kayit_yolu, temizlenmis)
        print(f"{dosya_adi} işlendi ve kaydedildi.")
