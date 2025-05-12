import os
import pandas as pd
import shutil

# Klasör yolları
giris_klasoru = r"C:\Users\atade\Desktop\veri_seti\Tüm_csv"       # CSV dosyalarının bulunduğu klasör
cikis_klasoru = r"C:\Users\atade\Desktop\veri_seti\dosya"  # Uygun dosyaların kopyalanacağı klasör

# Çıkış klasörü yoksa oluştur
os.makedirs(cikis_klasoru, exist_ok=True)

# Tüm CSV dosyalarını işle
for dosya_adi in os.listdir(giris_klasoru):
    if dosya_adi.endswith('.csv'):
        dosya_yolu = os.path.join(giris_klasoru, dosya_adi)
        
        try:
            # Dosyayı oku (başlık satırı atlanır)
            df = pd.read_csv(dosya_yolu, skiprows=1, header=None)
            frekans = df[0]
            db_degerleri = df[1]
            
            # Global minimum dB değerinin konumu
            min_index = db_degerleri.idxmin()
            
            # Global minimumdan sonraki dB değerlerini al
            sonrasi_db = db_degerleri[min_index:]
            
            # Maksimum değeri kontrol et
            max_sonrasi = sonrasi_db.max()
            if -2 <= max_sonrasi <= 0:
                # Dosyayı kopyala
                hedef_yol = os.path.join(cikis_klasoru, dosya_adi)
                shutil.copy2(dosya_yolu, hedef_yol)
                print(f"{dosya_adi} kopyalandı.")
        
        except Exception as e:
            print(f"{dosya_adi} işlenirken hata oluştu: {e}")
