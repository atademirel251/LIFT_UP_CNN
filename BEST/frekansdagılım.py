import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import tensorflow as tf
from keras.models import load_model, Model
from keras import layers
from keras.utils import load_img, img_to_array
from keras.optimizers import Adam
from keras.utils.generic_utils import get_custom_objects
import pandas as pd
from natsort import natsorted

# ===================== CUSTOM LOSS FUNCTIONS =====================
def diagonal_symmetry_loss(y_pred):
    """
    Calculates diagonal symmetry loss for the generated images
    """
    y_pred_transposed = tf.transpose(y_pred, perm=[0, 2, 1, 3])
    symmetry_diff = tf.abs(y_pred - y_pred_transposed)
    height = tf.shape(y_pred)[1]
    mask = 1.0 - 0.5 * tf.eye(height)
    mask = tf.expand_dims(mask, axis=0)
    mask = tf.expand_dims(mask, axis=-1)
    weighted_diff = symmetry_diff * mask
    return tf.reduce_mean(weighted_diff)

def block_homogeneity_loss(y_pred, block_size=8):
    """
    Encourages 8x8 blocks to be homogeneous (either black or white)
    """
    # Convert to grayscale if RGB
    if y_pred.shape[-1] == 3:
        y_pred = tf.reduce_mean(y_pred, axis=-1, keepdims=True)
    
    batch_size = tf.shape(y_pred)[0]
    height = tf.shape(y_pred)[1]
    width = tf.shape(y_pred)[2]
    
    # Reshape into blocks
    blocks = tf.reshape(y_pred, 
                       [batch_size, height//block_size, block_size, 
                        width//block_size, block_size, 1])
    blocks = tf.transpose(blocks, [0, 1, 3, 2, 4, 5])
    
    # Calculate variance for each block
    block_var = tf.math.reduce_variance(blocks, axis=[3,4,5])
    
    return tf.reduce_mean(block_var)

# Register custom losses
get_custom_objects().update({
    "diagonal_symmetry_loss": diagonal_symmetry_loss,
    "block_homogeneity_loss": block_homogeneity_loss
})

# ===================== IMAGE LOADING AND PROCESSING =====================
def load_images_and_frequencies(image_folder, csv_folder, img_size=(64, 64)):
    images = []
    frequencies = []
    
    csv_files = natsorted([f for f in os.listdir(csv_folder) if f.endswith(".csv")])
    
    for csv_file in csv_files:
        csv_path = os.path.join(csv_folder, csv_file)
        df = pd.read_csv(csv_path)
        
        if df.shape[1] >= 2:
            min_dB_index = df.iloc[:, 1].idxmin()
            freq_value = df.iloc[min_dB_index, 0]
            frequencies.append(int(round(freq_value)))
        else:
            print(f"Hatalı CSV formatı: {csv_file}. Atlanıyor.")
    
    image_files = natsorted([f for f in os.listdir(image_folder) if f.endswith((".png", ".jpg"))])
    
    for img_file in image_files:
        img_path = os.path.join(image_folder, img_file)
        img = load_img(img_path)
        img = img_to_array(img)
        height, width, _ = img.shape
        img = img[:height//2, :width//2, :]  # Crop to quarter
        img = tf.image.resize(img, img_size)
        img = (img - 127.5) / 127.5  # Normalize to [-1, 1]
        images.append(img)
    
    print(f"Yüklenen resim sayısı: {len(images)}, frekans sayısı: {len(frequencies)}")
    print(f"Örnek frekans değerleri: {np.unique(frequencies)}")
    return np.array(images), np.array(frequencies)

# ===================== MODEL LOADING =====================
def load_generator_model(model_path):
    try:
        return load_model(model_path, compile=False)
    except Exception as e:
        print(f"Model yükleme hatası: {e}")
        # Ensure custom objects are registered
        get_custom_objects().update({
            "diagonal_symmetry_loss": diagonal_symmetry_loss,
            "block_homogeneity_loss": block_homogeneity_loss
        })
        return load_model(model_path, compile=False)





import matplotlib.pyplot as plt
import pandas as pd

# Frekansları al
image_folder = r"C:\Users\atade\Desktop\14348_VERi\resim128_NET"
csv_folder = r"C:\Users\atade\Desktop\14348_VERi\Tüm_csv"

_, frequencies = load_images_and_frequencies(image_folder, csv_folder)

# Pandas ile frekans dağılımını al
freq_series = pd.Series(frequencies)
freq_counts = freq_series.value_counts().sort_index()

# Tablo olarak yazdır
print("Frekans Dağılımı:")
print(freq_counts)

# Matplotlib ile çubuk grafik çiz
plt.figure(figsize=(12, 6))
plt.bar(freq_counts.index, freq_counts.values, color='skyblue')
plt.title("Frekanslara Göre Örnek Sayısı Dağılımı")
plt.xlabel("Frekans Değeri")
plt.ylabel("Örnek Sayısı")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
