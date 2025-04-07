
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

# Load images and frequencies
image_folder = r"C:\Users\atade\Desktop\13579_veri\resim128_opencv_rgb"
csv_folder = r"C:\Users\atade\Desktop\13579_veri\Tüm_csv"
images, frequencies = load_images_and_frequencies(image_folder, csv_folder)

# Normalize frequencies to [-1, 1]
frequencies = (frequencies - np.mean(frequencies)) / np.std(frequencies) 
# Kaydedilen generator modelini yükle
generator = load_model(r"C:\Users\atade\Desktop\LIFT_UP_CNN\generator_model4.h5")



def generate_image_for_frequency(generator, frequency, latent_dim, mean, std):
    # Frekansı normalize et
    normalized_frequency = (frequency - mean) / std
    
    # Gürültü vektörünü ve normalize edilmiş frekansı generator'a giriş olarak ver
    noise = np.random.normal(0, 1, (1, latent_dim))
    frequency = np.array([normalized_frequency])
    
    # Görüntüyü üret
    generated_image = generator.predict([noise, frequency])
    
    # Görüntüyü [-1, 1] aralığından [0, 1] aralığına getir
    generated_image = 0.5 * generated_image + 0.5
    
    return generated_image[0]



    # Eğitim sırasında kullanılan frekansların ortalama ve standart sapmasını yükle
mean_frequency = np.mean(frequencies)
std_frequency = np.std(frequencies)

# Örnek frekans değeri ile görüntü üret
frequency = 15
latent_dim =32# Örneğin, 10 GHz



# Örnek frekans değeri ile görüntü üret
generated_image = generate_image_for_frequency(generator, frequency, latent_dim, mean_frequency, std_frequency)

# Eksik pikselleri doldur


# Doldurulmuş resmi göster
plt.imshow(generated_image, interpolation='nearest')
plt.axis('off')
plt.show()
print("Inpainted image shape:", generated_image.shape)

# Doldurulmuş görüntüyü kaydet
output_path_inpainted = r"C:\Users\atade\Desktop\image_rgb1.png"
plt.imshow(generated_image, interpolation='nearest')
plt.axis('off')
plt.savefig(output_path_inpainted, bbox_inches='tight', pad_inches=0)
print(f"Inpainted image saved to: {output_path_inpainted}")

