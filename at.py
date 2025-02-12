import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.callbacks import EarlyStopping

# GPU Belleği Yönetimi
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Veri Hazırlığı (Normalizasyon ve PCA ile)
def load_data_in_order(image_folder, csv_folder, max_length=102):
    image_files = sorted([f for f in os.listdir(image_folder) if not f.startswith('.')], key=str.lower)
    csv_files = sorted([f for f in os.listdir(csv_folder) if not f.startswith('.') and not f.endswith('.ipynb_checkpoints')], key=str.lower)

    if len(image_files) != len(csv_files):
        raise ValueError("Görüntü ve CSV dosyalarının sayısı eşleşmiyor!")

    images = []
    outputs = []
    
    for img_file, csv_file in zip(image_files, csv_files):
        image_path = os.path.join(image_folder, img_file)
        image = Image.open(image_path)
        image_array = np.array(image) / 255.0
        images.append(image_array)

        csv_path = os.path.join(csv_folder, csv_file)
        csv_data = pd.read_csv(csv_path, usecols=[0, 1], skiprows=1, header=None).values
        freq = csv_data[:, 0]
        s21 = csv_data[:, 1]
        combined = np.column_stack((freq, s21))

        if len(combined) > max_length:
            combined = combined[:max_length]
        elif len(combined) < max_length:
            pad = np.zeros((max_length - len(combined), 2))
            combined = np.vstack((combined, pad))

        outputs.append(combined)

    images = np.array(images, dtype=np.float32)
    outputs = np.array(outputs, dtype=np.float32)

    return images, outputs

# PCA Uygulama ve Tersine Çevirme
class PCAWrapper:
    def __init__(self, n_components=None):
        self.pca = None
        self.n_components = n_components
    
    def fit_transform(self, data):
        n_samples = data.shape[0]
        data_reshaped = data.reshape(n_samples, -1)
        self.pca = PCA(n_components=self.n_components).fit(data_reshaped)
        return self.pca.transform(data_reshaped)
    
    def inverse_transform(self, data):
        return self.pca.inverse_transform(data).reshape(-1, 102, 2)

# Veri klasörleri
image_folder = "C:/Users/atade/Desktop/1000verısetı/input_Resşm"
csv_folder = "C:/Users/atade/Desktop/1000verısetı/csv"

# Veriyi yükle
images, s21_params = load_data_in_order(image_folder, csv_folder, max_length=102)

# Veriyi eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(images, s21_params, test_size=0.2, random_state=42)

# PCA uygula
pca_wrapper = PCAWrapper(n_components=10)
y_train_pca = pca_wrapper.fit_transform(y_train)
y_test_pca = pca_wrapper.fit_transform(y_test)

# Pre-trained VGG16 Modeli ile Özellik Çıkartma
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False

model_input = base_model.input
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.4)(x)
output = Dense(10, activation='linear')(x)

model = Model(inputs=model_input, outputs=output)

# Öğrenme Hızı Planlayıcı
decay_steps = len(X_train) // 16
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=decay_steps,
    decay_rate=0.9
)

# Modeli derle
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
              loss='mean_squared_error',
              metrics=['mae'])

# Early Stopping Callback
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Modeli eğitme
history = model.fit(
    X_train, y_train_pca,
    epochs=40,
    batch_size=16,
    validation_data=(X_test, y_test_pca),
    callbacks=[early_stopping],
    verbose=1
)

# Eğitim Kaybı Grafiği
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Eğitim Kaybı', linestyle='-', marker='o', alpha=0.7)
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı', linestyle='--', marker='x', alpha=0.7)
plt.xlabel("Epochs")
plt.ylabel("Kayıp (Loss)")
plt.title("Eğitim ve Doğrulama Kaybı Grafiği")
plt.legend()
plt.grid(True)
plt.show()

# Test ve Tahmin Grafiği
example_index = 1
example_input = X_test[example_index]
example_output = y_test[example_index]

predicted_pca_output = model.predict(example_input[np.newaxis, ...])[0]
predicted_output = pca_wrapper.inverse_transform(predicted_pca_output[np.newaxis, :])[0]

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.imshow(example_input.squeeze(), cmap='gray')
plt.title("Test Girdisi (Geometrik Desen)")
plt.axis('off')

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

# Modeli kaydet
model.save("C:/Users/atade/Desktop/test_sonuçları/VGG16+TEST/model/VGG16_PCA_model.keras")
print("Model '.keras' formatında kaydedildi.")
