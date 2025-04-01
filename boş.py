import os
import cv2

# Klasör yolu
input_folder = r"C:\Users\atade\Desktop\Sinan_Veriler\resim2"
output_folder = os.path.join(input_folder, "Resized")  # Yeni klasör oluştur

# Eğer çıktı klasörü yoksa oluştur
os.makedirs(output_folder, exist_ok=True)

# Klasördeki tüm resimleri işle
for filename in os.listdir(input_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):  # Resim uzantılarını kontrol et
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)  # Resmi oku

        if img is None:
            print(f"❌ Hata: {filename} okunamadı!")
            continue

        # Resmi 128x128 boyutuna getir
        resized_img = cv2.resize(img, (128, 128))

        # Yeni yolu belirle ve kaydet
        save_path = os.path.join(output_folder, filename)
        cv2.imwrite(save_path, resized_img)

        print(f"✅ {filename} başarıyla yeniden boyutlandırıldı ve kaydedildi.")

print("✅ Tüm görüntüler işlendi ve 128x128 boyutuna getirildi!")
