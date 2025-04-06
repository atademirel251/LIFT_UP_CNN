import os
import pandas as pd
from natsort import natsorted  # Doğal sıralama için

# CSV dosyalarının bulunduğu klasörün path'ini belirtin
csv_folder = r"C:\Users\atade\Desktop\EREN_Veriler\e7\veriler\92\csv"

# Gizli dosyaları ve istenmeyen dosyaları filtrele
csv_files = natsorted(
    [f for f in os.listdir(csv_folder) if f.endswith('.csv')],
    key=lambda x: x.lower()  # Büyük/küçük harf duyarlılığını kaldır
)

# Dosya adlarını değiştir ve Excel'e kaydet
data = []  # Eski ve yeni adları saklamak için

for index, old_name in enumerate(csv_files, start=1):
    # Yeni dosya adını oluştur (ATA1.csv, ATA2.csv, ...)
    new_name = f"Eren{index}.csv"

    # Dosyayı yeniden adlandır
    old_path = os.path.join(csv_folder, old_name)
    new_path = os.path.join(csv_folder, new_name)
    os.rename(old_path, new_path)

    # Eski ve yeni adları listeye ekle
    data.append({"Eski Ad": old_name, "Yeni Ad": new_name})

# DataFrame oluştur
df = pd.DataFrame(data)

# Excel dosyası olarak kaydet


