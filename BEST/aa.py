import cv2
import os

# İşlenecek klasör yolu
klasor_yolu = r"C:\Users\atade\Desktop\s11_dataset\img"

# Geçerli uzantılar
gecerli_uzantilar = ['.jpg', '.jpeg', '.png']

# Tüm dosyaları al
for dosya in os.listdir(klasor_yolu):
    dosya_yolu = os.path.join(klasor_yolu, dosya)

    # Dosya mı ve uzantı uygun mu?
    if os.path.isfile(dosya_yolu) and os.path.splitext(dosya)[1].lower() in gecerli_uzantilar:
        # Gri olarak oku
        gri_resim = cv2.imread(dosya_yolu, cv2.IMREAD_GRAYSCALE)

        # Zaten renkli ise atla (isteğe bağlı)
        if len(gri_resim.shape) == 2:  # Tek kanallıysa
            # 3 kanallıya çevir
            resim_3kanal = cv2.cvtColor(gri_resim, cv2.COLOR_GRAY2BGR)

            # Üzerine kaydet
            cv2.imwrite(dosya_yolu, resim_3kanal)
            print(f"{dosya} dosyası 3 kanallı hale getirildi.")
        else:
            print(f"{dosya} zaten 3 kanallı, atlandı.")
