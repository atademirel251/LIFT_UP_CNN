import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
from natsort import natsorted 


# GPU Bellek Yönetimi
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Ağırlıklı Kayıp Fonksiyonu (Dip Noktalara Önem Ver)
def weighted_loss(y_true, y_pred):
    error = K.abs(y_true - y_pred)
    weight = K.exp(-0.07 * K.abs(y_true))  
    dy_dx_true = K.abs(y_true[:, 1:] - y_true[:, :-1])
    dy_dx_pred = K.abs(y_pred[:, 1:] - y_pred[:, :-1])
    gradient_penalty = K.mean(K.abs(dy_dx_true - dy_dx_pred))
    loss = K.mean(weight * error) + (0.2 * gradient_penalty)
    return loss 




def load_data_in_order(image_folder, csv_folder, max_length=101, local_min_threshold=0.2):
    """ image_files = sorted([f for f in os.listdir(image_folder) if not f.startswith('.')], key=str.lower)
    csv_files = sorted([f for f in os.listdir(csv_folder) if not f.startswith('.') and not f.endswith('.ipynb_checkpoints')], key=str.lower) """
    image_files = natsorted(
    [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))],
    key=lambda x: x.lower()  # Büyük/küçük harf duyarlılığını kaldır
)
    csv_files = natsorted(
    [f for f in os.listdir(csv_folder) if f.endswith('.csv')],
    key=lambda x: x.lower()  # Büyük/küçük harf duyarlılığını kaldır
)
    if len(image_files) != len(csv_files):
        raise ValueError("Görüntü ve CSV dosyalarının sayısı eşleşmiyor!")

    images_original, images_edited, outputs = [], [], []
    for img_file, csv_file in zip(image_files, csv_files):
        image_path = os.path.join(image_folder, img_file)
        image = Image.open(image_path)  # Orijinal görüntüyü aç

        # **Orijinal görüntüyü kaydet**
        image_original = np.array(image) / 255.0
        images_original.append(image_original)

        # **Çeyrek bölgeyi al (sol üst köşe)**
        w, h = image.size
        quarter_image = image.crop((0, 0, w // 2, h // 2))

        # **Çeyrek bölgeyi 64x64 boyutuna küçült**
        resized_image = quarter_image.resize((64, 64))
        image_edited = np.array(resized_image) / 255.0
        images_edited.append(image_edited)

        # **CSV verisini al**
        csv_path = os.path.join(csv_folder, csv_file)
        csv_data = pd.read_csv(csv_path, usecols=[0, 1], skiprows=1, header=None).values
        combined = np.column_stack((csv_data[:, 0], csv_data[:, 1]))

        # **Yerel minimumları bul ve interpolasyon yap**
        x = combined[:, 0]
        y = combined[:, 1]

        local_min_indices = np.where(np.diff(np.sign(np.diff(y))) > 0)[0] + 1
        
        for idx in local_min_indices:
            if idx > 0 and idx < len(x) - 1:
                local_x = x[max(0, idx-6):min(len(x), idx+6)]
                local_y = y[max(0, idx-6):min(len(x), idx+6)]

                interp_x = np.linspace(local_x[0], local_x[-1], 7)
                interp_y = np.interp(interp_x, local_x, local_y)

                new_data = np.column_stack((interp_x, interp_y))
                combined = np.vstack((combined[:idx-1], new_data, combined[idx+2:]))

        # max_length'e göre kısıtlama
        if len(combined) > max_length:
            combined = combined[:max_length]
        elif len(combined) < max_length:
            pad = np.zeros((max_length - len(combined), 2))
            combined = np.vstack((combined, pad))

        outputs.append(combined)

    return np.array(images_original, dtype=np.float32), np.array(images_edited, dtype=np.float32), np.array(outputs, dtype=np.float32)
# Veri klasörleri
image_folder = r"C:\Users\atade\Desktop\10440_veri\input_Resim"
csv_folder = r"C:\Users\atade\Desktop\10440_veri\csv"

# **Yeni veri yükleme fonksiyonu çağırılıyor**
images_original, images_edited, s21_params = load_data_in_order(image_folder, csv_folder, max_length=101)

# **Model sadece çeyrek bölgeyi (edited) kullanıyor**
X_train, X_test, y_train, y_test = train_test_split(images_edited, s21_params, test_size=0.2, random_state=42)


# Modeli yükle
model_path = r"C:\Users\atade\Desktop\test_sonuçları\VGG16+TEST\model\Yeni10441_64x64+15katman+ınterpolatıon7.keras"
model = tf.keras.models.load_model(model_path, custom_objects={"weighted_loss": weighted_loss})

# Modeli kullanarak tahmin yap
y_pred = model.predict(X_test).reshape(y_test.shape)

# MAPE hesaplama fonksiyonu
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

# Test seti üzerinde MAPE hesapla
mape_score = mean_absolute_percentage_error(y_test, y_pred)
print(f"Test Seti İçin MAPE: {mape_score:.2f}%") 


# **Kullanıcıya 18. indeksin ORİJİNAL görüntüsünü göster**
example_index = 449
example_original = images_original[example_index]  # Orijinal görüntü (128x128)
example_input = X_test[example_index]  # Modelin kullandığı görüntü (64x64)
example_output = y_test[example_index]
predicted_output = y_pred[example_index]


plt.figure(figsize=(14, 6))

# **Orijinal görüntüyü göster**
plt.subplot(1, 2, 1)
plt.imshow(example_original.squeeze(), cmap='gray')
plt.title(" Orijinal Geometrik Desen")
plt.axis('off')

# **Tahmini vs Gerçek Değeri Çiz**
plt.subplot(1, 2, 2)
plt.plot(example_output[:, 0], example_output[:, 1], label="Gerçek Değer", linestyle='none', marker='o', alpha=0.7)
plt.plot(predicted_output[:, 0], predicted_output[:, 1], label="Tahmin Değer", linestyle='none', marker='x', alpha=0.7)
plt.xlabel("Frekans (GHz)")
plt.ylabel("S21 Parametre Değeri")
plt.title("Gerçek ve Tahmini S21 Grafiği")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
