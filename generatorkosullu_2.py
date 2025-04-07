import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import os
from keras.models import load_model
import numpy as np
import cv2
from scipy.ndimage import binary_fill_holes
import os
import numpy as np
import tensorflow as tf
from keras import layers
from keras.utils import load_img, img_to_array
from keras.models import Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
from natsort import natsorted
def load_images_and_frequencies(image_folder, csv_folder, img_size=(64, 64)):
    images = []
    frequencies = []
    
    # CSV dosyalarını doğal sıralama ile oku
    csv_files = natsorted([f for f in os.listdir(csv_folder) if f.endswith(".csv")])
    
    for csv_file in csv_files:
        # CSV'den frekansı oku ve integer'a yuvarla
        csv_path = os.path.join(csv_folder, csv_file)
        df = pd.read_csv(csv_path)
        
        if df.shape[1] >= 2:
            min_dB_index = df.iloc[:, 1].idxmin()
            freq_value = df.iloc[min_dB_index, 0]
            frequencies.append(int(round(freq_value)))  # Yuvarlama ve integer dönüşüm
        else:
            print(f"Hatalı CSV formatı: {csv_file}. Atlanıyor.")
    
    # Resimleri aynı sırayla yükle
    image_files = natsorted([f for f in os.listdir(image_folder) if f.endswith((".png", ".jpg"))])
    
    for img_file in image_files:
        img_path = os.path.join(image_folder, img_file)
        img = load_img(img_path)
        img = img_to_array(img)
        height, width, _ = img.shape
        #img = img[height//2:, width//2:, :]
        img = img[:height//2, :width//2, :]   # Crop the image
        img = tf.image.resize(img, img_size)  # Resize the image
        img = (img - 127.5) / 127.5  # Normalize to [-1, 1]
        images.append(img)
    
    print(f"Yüklenen resim sayısı: {len(images)}, frekans sayısı: {len(frequencies)}")
    print(f"Örnek frekans değerleri: {np.unique(frequencies)}")  # Benzersiz frekansları göster
    return np.array(images), np.array(frequencies)
# Kayıtlı modeli yükle
try:
    generator = load_model("generator_model4.h5", compile=False)
except:
    from keras.utils.generic_utils import get_custom_objects
    get_custom_objects().update({"diagonal_symmetry_loss": diagonal_symmetry_loss})
    generator = load_model("generator_model4.h5", compile=False)

# Orijinal frekans istatistikleri
def get_frequency_stats():
    image_folder = r"C:\Users\atade\Desktop\13579_veri\resim128_opencv_rgb"
    csv_folder = r"C:\Users\atade\Desktop\13579_veri\Tüm_csv"
    _, frequencies = load_images_and_frequencies(image_folder, csv_folder)
    return np.mean(frequencies), np.std(frequencies)

if not hasattr(get_frequency_stats, 'freq_mean'):
    get_frequency_stats.freq_mean, get_frequency_stats.freq_std = get_frequency_stats()

def generate_quarter_pattern(frequency, latent_dim=32):
    """Sadece sol üst çeyrek deseni üretir"""
    normalized_freq = (frequency - get_frequency_stats.freq_mean) / get_frequency_stats.freq_std
    noise = np.random.normal(0, 1, (1, latent_dim))
    generated_image = generator.predict([noise, np.array([[normalized_freq]])])
    return 0.5 * generated_image[0] + 0.5  # [0,1] aralığına dönüştür

def save_quarter_pattern(pattern, output_path, dpi=300):
    """Çeyrek deseni kaydeder"""
    plt.figure(figsize=(5, 5))  # Daha küçük boyut
    plt.imshow(pattern, interpolation='nearest')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=dpi)
    plt.close()

def generate_and_save_quarter(frequency, output_dir="quarter_patterns"):
    """
    Sadece sol üst çeyrek deseni üretir ve kaydeder
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, f"quarter_pattern_freq_{frequency}.png")
    quarter_pattern = generate_quarter_pattern(frequency)
    save_quarter_pattern(quarter_pattern, output_path)
    print(f"{frequency} frekansı için çeyrek desen {output_path} dosyasına kaydedildi.")

# Kullanım örnekleri:
if __name__ == "__main__":
    # Tek bir frekans için
    generate_and_save_quarter(10)
    
    # 6-20 arası tüm frekanslar için
    for freq in range(6, 21):
        generate_and_save_quarter(freq)
    
    # Özel klasör için
    generate_and_save_quarter(12, "custom_quarters")