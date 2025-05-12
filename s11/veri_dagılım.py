import os
import csv
from collections import Counter

# Klasör ve çıktı dosyası
csv_klasoru = r"C:\Users\atade\Desktop\veri_seti\s11_csv"
output_dosyasi = r"C:\Users\atade\Desktop\LIFT_UP_CNN\generated_patterns\Yeni Text Document.txt"

# Frekansları toplayacağımız liste
frekans_listesi = []

for dosya_adi in os.listdir(csv_klasoru):
    if dosya_adi.endswith(".csv"):
        dosya_yolu = os.path.join(csv_klasoru, dosya_adi)

        with open(dosya_yolu, newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Başlığı atla

            frekanslar = []
            db_degerleri = []

            for satir in reader:
                try:
                    frekans = float(satir[0])
                    db = float(satir[1])
                    frekanslar.append(frekans)
                    db_degerleri.append(db)
                except:
                    continue

            if not db_degerleri:
                continue

            # Dip noktasından sonra maksimum dB değeri
            dip_index = db_degerleri.index(min(db_degerleri))
            sonraki_db = db_degerleri[dip_index+1:]
            sonraki_frekans = frekanslar[dip_index+1:]

            if not sonraki_db:
                continue

            max_index = sonraki_db.index(max(sonraki_db))
            hedef_frekans = int(round(sonraki_frekans[max_index]))
            frekans_listesi.append(hedef_frekans)

# Frekansları say
frekans_sayaci = Counter(frekans_listesi)

# Sayılarını sıraya koy
sirali = sorted(frekans_sayaci.items())

# Dosyaya yaz
with open(output_dosyasi, "w") as f:
    f.write(f"Toplam {len(frekans_listesi)} dosya işlendi.\n\n")
    for frekans, adet in sirali:
        f.write(f"{frekans} Hz: {adet} kez\n")

print(f"{len(frekans_listesi)} dosya işlendi. Sonuçlar '{output_dosyasi}' dosyasına yazıldı.")
