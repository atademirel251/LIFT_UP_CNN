import os
import pandas as pd
import matplotlib.pyplot as plt

# CSV dosyalarının bulunduğu klasör
csv_klasoru = r"C:\Users\atade\Desktop\veri_seti\s11_csv"  # klasör adı veya tam yol
output_klasoru = os.path.join(csv_klasoru, "grafikler10")

# Çıktı klasörü yoksa oluştur
os.makedirs(output_klasoru, exist_ok=True)

# Klasördeki tüm CSV dosyalarını işle
for dosya_adi in os.listdir(csv_klasoru):
    if dosya_adi.endswith(".csv"):
        dosya_yolu = os.path.join(csv_klasoru, dosya_adi)
        
        # CSV'yi oku
        try:
            df = pd.read_csv(dosya_yolu, header=0)
            frekans = df.iloc[:, 0]
            dB = df.iloc[:, 1]
            
            # Grafik oluştur
            plt.figure()
            plt.plot(frekans, dB, label=dosya_adi)
            plt.xlabel("Frekans (Hz)")
            plt.ylabel("S11 (dB)")
            plt.title(f"S11 Grafiği - {dosya_adi}")
            plt.grid(True)
            plt.legend()

            # Grafik dosya adı
            grafik_dosya_adi = os.path.splitext(dosya_adi)[0] + ".png"
            grafik_kayit_yolu = os.path.join(output_klasoru, grafik_dosya_adi)
            
            # Kaydet
            plt.savefig(grafik_kayit_yolu)
            plt.close()
            print(f"{grafik_dosya_adi} kaydedildi.")
        except Exception as e:
            print(f"{dosya_adi} dosyasında hata: {e}")
