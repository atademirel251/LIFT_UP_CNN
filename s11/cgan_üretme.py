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
import os
import pandas as pd
import numpy as np
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
            # 5'ten büyük frekansları filtrele
            filtered_df = df[df.iloc[:, 0] > 5].reset_index(drop=True)

            if not filtered_df.empty:
                db_values = filtered_df.iloc[:, 1]
                min_db_index = db_values.idxmin()

                # Minimum dB değerinden sonraki verileri al
                post_min_df = filtered_df.iloc[min_db_index + 1:]

                if not post_min_df.empty:
                    max_db_index = post_min_df.iloc[:, 1].idxmax()
                    freq_value = post_min_df.loc[max_db_index, post_min_df.columns[0]]
                    frequencies.append(round(freq_value))
                else:
                    print(f"{csv_file} içinde global min dB'den sonra veri yok. Atlanıyor.")
                    frequencies.append(0)
            else:
                print(f"{csv_file} içinde 5 üstü frekans bulunamadı. Atlanıyor.")
                frequencies.append(0)
        else:
            print(f"Hatalı CSV formatı: {csv_file}. Atlanıyor.")
            frequencies.append(0)

    # Resimleri aynı sırayla yükle
    image_files = natsorted([f for f in os.listdir(image_folder) if f.endswith((".png", ".jpg"))])
    
    for img_file in image_files:
        img_path = os.path.join(image_folder, img_file)
        img = load_img(img_path)
        img = img_to_array(img)
        height, width, _ = img.shape
        img = img[height//2:, width//2:, :]  # Resmi kırp

        img = tf.image.resize(img, img_size)  # Yeniden boyutlandır
        img = (img - 127.5) / 127.5  # Normalize et [-1, 1]
        images.append(img)
    
    print(f"Yüklenen resim sayısı: {len(images)}, frekans sayısı: {len(frequencies)}")

    if len(images) != len(frequencies):
        raise ValueError(f"Resim sayısı ({len(images)}) ve CSV sayısı ({len(frequencies)}) eşit değil!")

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

# ===================== PATTERN GENERATION =====================
class PatternGenerator:
    def __init__(self, model_path):
        self.generator = load_generator_model(model_path)
        self.latent_dim = 32
        self.freq_mean, self.freq_std = self._get_frequency_stats()
    
    def _get_frequency_stats(self):
        image_folder = r"C:\Users\atade\Desktop\veri_seti_s11\s11_resim"
        csv_folder = r"C:\Users\atade\Desktop\veri_seti_s11\s11_csv"
        _, frequencies = load_images_and_frequencies(image_folder, csv_folder)
        return np.mean(frequencies), np.std(frequencies)
    
    def generate_quarter_pattern(self, frequency):
        """Generates a single quarter pattern for given frequency"""
        normalized_freq = (frequency - self.freq_mean) / self.freq_std
        noise = np.random.normal(0, 1, (1, self.latent_dim))
        generated_image = self.generator.predict([noise, np.array([[normalized_freq]])], verbose=0)
        return 0.5 * generated_image[0] + 0.5  # Convert to [0,1] range
    
    def clean_8x8_blocks(self, image, threshold=0.5):
        """Post-processing to clean up 8x8 blocks"""
        h, w = image.shape[:2]
        cleaned = image.copy()
        
        for y in range(0, h, 8):
            for x in range(0, w, 8):
                block = image[y:y+8, x:x+8]
                avg = np.mean(block)
                cleaned[y:y+8, x:x+8] = 1 if avg > threshold else 0
        return cleaned
    
    def save_pattern(self, pattern, output_path, dpi=300):
        """Saves the generated pattern"""
        plt.figure(figsize=(5, 5))
        plt.imshow(pattern, interpolation='nearest', cmap='gray')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=dpi)
        plt.close()
    
    def generate_and_save(self, frequency, output_dir="generated_patterns"):
        """Full pipeline: generate, clean, and save pattern"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        output_path = os.path.join(output_dir, f"pattern_freq_{frequency}.png")
        pattern = self.generate_quarter_pattern(frequency)
        
        # Convert to grayscale and clean blocks
        pattern_gray = np.mean(pattern, axis=-1, keepdims=True)
        pattern_cleaned = self.clean_8x8_blocks(pattern_gray)
        
        self.save_pattern(pattern_cleaned, output_path)
        print(f"Frekans {frequency} için desen {output_path} kaydedildi.")

# ===================== MAIN EXECUTION =====================
if __name__ == "__main__":
    # Initialize with your model path
    MODEL_PATH = r"C:\Users\atade\Desktop\LIFT_UP_CNN\Frekanslıgenerator_model12_mayıs4_15.h5"
    
    generator = PatternGenerator(MODEL_PATH)
    
    # Generate for specific frequencies
    test_frequencies = range(8, 20)  # Or specific frequencies [6, 10, 15, etc.]
    
    for freq in test_frequencies:
        generator.generate_and_save(freq, "generated_patterns")
    
    print("Tüm desenler başarıyla oluşturuldu ve kaydedildi.")