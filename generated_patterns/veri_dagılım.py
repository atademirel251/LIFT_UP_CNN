import os
import pandas as pd
import numpy as np
from collections import Counter
from natsort import natsorted
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def load_images_and_frequencies(image_folder, csv_folder, img_size=(64, 64)):
    images = []
    frequencies = []

    csv_files = natsorted([f for f in os.listdir(csv_folder) if f.endswith(".csv")])
    
    for csv_file in csv_files:
        csv_path = os.path.join(csv_folder, csv_file)
        df = pd.read_csv(csv_path)

        if df.shape[1] >= 2:
            filtered_df = df[df.iloc[:, 0] > 5]  # Frekansı 5 GHz'den büyük olanlar

            if not filtered_df.empty:
                max_db_index = filtered_df.iloc[:, 1].idxmax()
                freq_value = df.iloc[max_db_index, 0]
                frequencies.append(int(freq_value))
            else:
                print(f"{csv_file} içinde 5 üstü frekans bulunamadı. Atlanıyor.")
                frequencies.append(0)
        else:
            print(f"Hatalı CSV formatı: {csv_file}. Atlanıyor.")

    # Frekans sayımını yap
    frequency_counts = Counter(frequencies)

    # Sayımı CSV dosyasına yaz
    count_df = pd.DataFrame(sorted(frequency_counts.items()), columns=["Frekans", "Desen Sayısı"])
    count_df.to_csv("frekans_sayim.csv", index=False)
    print("Frekans sayımı 'frekans_sayim.csv' dosyasına kaydedildi.")

    # Resimleri aynı sırayla yükle
    image_files = natsorted([f for f in os.listdir(image_folder) if f.endswith((".png", ".jpg"))])
    
    for img_file in image_files:
        img_path = os.path.join(image_folder, img_file)
        img = load_img(img_path)
        img = img_to_array(img)
        height, width, _ = img.shape
        img = img[height//2:, width//2:, :]  # Resmi kırp

        img = tf.image.resize(img, img_size)
        img = (img - 127.5) / 127.5  # Normalize et [-1, 1]
        images.append(img)
    
    print(f"Yüklenen resim sayısı: {len(images)}, frekans sayısı: {len(frequencies)}")

    if len(images) != len(frequencies):
        raise ValueError(f"Resim sayısı ({len(images)}) ve CSV sayısı ({len(frequencies)}) eşit değil!")

    print(f"Örnek frekans değerleri: {np.unique(frequencies)}")
    return np.array(images), np.array(frequencies)


# Load images and frequencies
image_folder = r"C:\Users\atade\Desktop\14348_VERi\resim128_NET"
csv_folder = r"C:\Users\atade\Desktop\14348_VERi\Tüm_csv"
images, frequencies = load_images_and_frequencies(image_folder, csv_folder)