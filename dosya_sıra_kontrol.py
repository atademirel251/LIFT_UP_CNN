import os
import pandas as pd
from natsort import natsorted  # natsort kütüphanesini ekleyin

# Klasörlerinizin path'lerini belirtin
image_folder = r"C:\Users\atade\Desktop\10440_veri\input_Resim"
csv_folder = r"C:\Users\atade\Desktop\10440_veri\csv"

# Gizli dosyaları ve istenmeyen dosyaları filtrele
image_files = natsorted(
    [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))],
    key=lambda x: x.lower()  # Büyük/küçük harf duyarlılığını kaldır
)
csv_files = natsorted(
    [f for f in os.listdir(csv_folder) if f.endswith('.csv')],
    key=lambda x: x.lower()  # Büyük/küçük harf duyarlılığını kaldır
)

# Dosya sayılarını yazdır
a = len(image_files)
b = len(csv_files)
print(f"Resim dosyaları: {a}, CSV dosyaları: {b}")

# Dosya adlarını karşılaştır
image_names = [os.path.splitext(f)[0] for f in image_files]
csv_names = [os.path.splitext(f)[0] for f in csv_files]

# Eşleşmeyen dosyaları bul
extra_csv_files = set(csv_names) - set(image_names)
extra_image_files = set(image_names) - set(csv_names)

if extra_csv_files:
    print(f"Eşleşmeyen CSV dosyaları: {extra_csv_files}")
if extra_image_files:
    print(f"Eşleşmeyen resim dosyaları: {extra_image_files}")

# Eşleşmeleri bir DataFrame'e aktar
if len(image_files) == len(csv_files):
    data = {
        "Image File": [os.path.join(image_folder, img) for img in image_files],
        "CSV File": [os.path.join(csv_folder, csv) for csv in csv_files]
    }
    df = pd.DataFrame(data)

    # Excel dosyası olarak kaydet
    output_excel_path = "output_matched_files9.xlsx"
    df.to_excel(output_excel_path, index=False)

    print(f"Eşleştirilmiş dosyalar '{output_excel_path}' dosyasına kaydedildi.")
else:
    print("Uyarı: Resim ve CSV dosyalarının sayısı eşit değil!")