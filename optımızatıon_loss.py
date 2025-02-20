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
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K

# GPU Bellek Yönetimi
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

 # Ağırlıklı Kayıp Fonksiyonu (Dip Noktalara Önem Ver)
def weighted_loss(y_true, y_pred):
    error = K.abs(y_true - y_pred)
    weight = K.exp(-0.05 * y_true)  # Küçük S21 değerlerine daha fazla ağırlık ver
    return K.mean(weight * error)   
 








""" def weighted_loss(y_true, y_pred):
    error = K.abs(y_true - y_pred)  # Mutlak hata hesaplanıyor

    # Dip noktalara daha fazla ağırlık veren fonksiyon
    weight = K.exp(-0.06 * K.abs(y_true))

    # Gradyan hesabı (yaklaşık türev)
    dy_dx = K.abs(y_pred[:, 1:] - y_pred[:, :-1])  # Fark alarak yaklaşık türev
    gradient_penalty = K.mean(dy_dx, axis=-1)  # Ortalama türev cezası hesapla

    # Toplam loss hesaplanıyor
    loss = K.mean(weight * error) + (0.1 * K.mean(gradient_penalty))  # λ=0.1 ağırlıklandırma
    return loss """




""" def weighted_loss(y_true, y_pred):
    error = K.abs(y_true - y_pred)
    alpha = 0.05  # Ağırlık katsayısı (ayar çekilebilir)

    # Dip ve tepe noktalarını cezalandıran ağırlık fonksiyonu
    weight = K.exp(-alpha * (y_true - K.mean(y_true))**2)  

    return K.mean(weight * error) """



""" def weighted_loss(y_true, y_pred):
    error = K.abs(y_true - y_pred)
    alpha = 0.01  # Küçük S21 değerlerine daha fazla, büyük S21'lere daha az ağırlık veren çarpan
    weight = K.exp(-alpha * K.abs(y_true))  
    return K.mean(weight * error) """




# Veri Hazırlığı (Normalizasyon ile)
def load_data_in_order(image_folder, csv_folder, max_length=101):
    image_files = sorted([f for f in os.listdir(image_folder) if not f.startswith('.')], key=str.lower)
    csv_files = sorted([f for f in os.listdir(csv_folder) if not f.startswith('.') and not f.endswith('.ipynb_checkpoints')], key=str.lower)
    print(f"Input resim sayısı: {len(image_files)}")
    print(f"csv sayısı: {len(csv_files)}")
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

# Veri klasörleri
image_folder = "C:/Users/atade/Desktop/1000verısetı/input_Resşm"
csv_folder = "C:/Users/atade/Desktop/1000verısetı/csv"

# Veriyi yükle
images, s21_params = load_data_in_order(image_folder, csv_folder, max_length=101)

# Veriyi eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(images, s21_params, test_size=0.2, random_state=42)

# Pre-trained VGG16 Modeli
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False  

model_input = base_model.input
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.4)(x)
output = Dense(s21_params.shape[1] * s21_params.shape[2], activation='linear')(x)

model = Model(inputs=model_input, outputs=output)

# Öğrenme Hızı Planlayıcı
decay_steps = len(X_train) // 16 
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=decay_steps,
    decay_rate=0.9
)
 # Modeli derle (Ağırlıklı Kayıp Fonksiyonuyla)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
              loss=weighted_derivative_loss,
              metrics=['mae']) 

""" model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=custom_loss,
              metrics=['mae']) """

# Veriyi uygun şekilde düzleştir
y_train_flat = y_train.reshape(y_train.shape[0], -1)
y_test_flat = y_test.reshape(y_test.shape[0], -1)

# Early Stopping
early_stopping = EarlyStopping(
    monitor='val_loss',  
    patience=10,         
    restore_best_weights=True  
)

# Modeli Eğitme
history = model.fit(
    X_train, y_train_flat,
    epochs=100,
    batch_size=16,
    validation_data=(X_test, y_test_flat),
    callbacks=[early_stopping],
    verbose=1
)

# Eğitim ve Doğrulama Kaybı Grafiği
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
example_index = 15
example_input = X_test[example_index]
example_output = y_test[example_index]

predicted_output = model.predict(example_input[np.newaxis, ...])[0].reshape(-1, 2)

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
model.save("C:/Users/atade/Desktop/test_sonuçları/VGG16+TEST/model/VGG16_feature_extracted_model_Yeni2000_turevliepoch.keras")
print("Model '.keras' formatında kaydedildi.")
