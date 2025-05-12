import os
import shutil

# Ayarlar
txt_dosya_yolu = r"C:\Users\atade\Desktop\veri_seti\aa.txt"   # İsimlerin yazılı olduğu .txt dosyası
giris_klasoru = r"C:\Users\atade\Desktop\veri_seti\resim128_NET"       # .png dosyalarının bulunduğu klasör
hedef_klasor = r"C:\Users\atade\Desktop\veri_seti\s11_resim"        # Dosyaların taşınacağı klasör

# Hedef klasör yoksa oluştur
os.makedirs(hedef_klasor, exist_ok=True)

# TXT dosyasını oku ve uzantıları .csv -> .png olarak değiştir
with open(txt_dosya_yolu, 'r') as f:
    dosya_isimleri = [line.strip().replace('.csv', '.png') for line in f]

# Dosyaları taşı
for dosya_adi in dosya_isimleri:
    kaynak_yol = os.path.join(giris_klasoru, dosya_adi)
    hedef_yol = os.path.join(hedef_klasor, dosya_adi)

    if os.path.exists(kaynak_yol):
        shutil.move(kaynak_yol, hedef_yol)
        print(f"{dosya_adi} taşındı.")
    else:
        print(f"{dosya_adi} bulunamadı.")

