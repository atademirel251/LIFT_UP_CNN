import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image
#ATA DEGISIKLIK

# 1. Veri Hazırlığı (Normalizasyon olmadan)
def load_data_in_order(image_folder, csv_folder, max_length=210):  # max_length = 210 olarak değiştirildi
    image_files = sorted([f for f in os.listdir(image_folder) if not f.startswith('.')])
    csv_files = sorted([f for f in os.listdir(csv_folder) if not f.startswith('.') and not f.endswith('.ipynb_checkpoints')])

    if len(image_files) != len(csv_files):
        raise ValueError("Görüntü ve CSV dosyalarının sayısı eşleşmiyor!")

    images = []
    outputs = []

    for img_file, csv_file in zip(image_files, csv_files):
        image_path = os.path.join(image_folder, img_file)
        image = Image.open(image_path)
        image_array = np.array(image)/255.0  # Görüntü üzerinde normalizasyon yok
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

    images = np.expand_dims(np.array(images, dtype=np.float32), axis=-1)
    outputs = np.array(outputs, dtype=np.float32)

    return images, outputs

image_folder = "/content/sample_data/output_resimler"
csv_folder = "/content/sample_data/output_CSV"

# max_length = 210
images, s21_params = load_data_in_order(image_folder, csv_folder, max_length=210)
X_train, X_test, y_train, y_test = train_test_split(images, s21_params, test_size=0.2, random_state=42)

# 2. Model Mimarisi
model = Sequential([
    Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),
    Flatten(),
    Dense(1024, activation='relu'),
    Dropout(0.4),
    Dense(s21_params.shape[1] * s21_params.shape[2], activation='linear')
])



# Öğrenme Hızı Planlayıcı
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=1000,
    decay_rate=0.9
)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
              loss='mean_squared_error',
              metrics=['mae'])

y_train_flat = y_train.reshape(y_train.shape[0], -1)
y_test_flat = y_test.reshape(y_test.shape[0], -1)

# 3. Model Eğitimi
history = model.fit(X_train, y_train_flat, epochs=100, batch_size=64, validation_split=0.2, verbose=1)

# 4. Eğitim ve Doğrulama Kaybı Grafiği
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Eğitim Kaybı', linestyle='-', marker='o', alpha=0.7)
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı', linestyle='--', marker='x', alpha=0.7)
plt.xlabel("Epochs")
plt.ylabel("Kayıp (Loss)")
plt.title("Eğitim ve Doğrulama Kaybı Grafiği")
plt.legend()
plt.grid(True)
plt.show()

# 5. Test ve Tahmin Grafiği
example_index = 1
example_input = X_test[example_index]
example_output = y_test[example_index]

predicted_output = model.predict(example_input[np.newaxis, ...])[0].reshape(-1, 2)

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.imshow(example_input.squeeze(), cmap='gray')
plt.title("Test Girdisi (Geometrik Desen)")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.plot(example_output[:, 0], example_output[:, 1], label="Gerçek Değer", linestyle='-', marker='o', alpha=0.7)
plt.plot(predicted_output[:, 0], predicted_output[:, 1], label="Tahmin Değer", linestyle='--', marker='x', alpha=0.7)
plt.xlabel("Frekans (GHz)")
plt.ylabel("S21 Parametre Değeri")
plt.title("Gerçek ve Tahmini S21 Grafiği")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Modeli kaydet
model.save("/content/sample_data/CNN_155(1)_non_normalized.keras")
print("Model '.keras' formatında kaydedildi.")
