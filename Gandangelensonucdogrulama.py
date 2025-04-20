import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K

# GPU Ayarları
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Özel Kayıp Fonksiyonu
def weighted_loss(y_true, y_pred):
    error = K.abs(y_true - y_pred)
    weight = K.exp(-0.07 * K.abs(y_true))  
    dy_dx_true = K.abs(y_true[:, 1:] - y_true[:, :-1])
    dy_dx_pred = K.abs(y_pred[:, 1:] - y_pred[:, :-1])
    gradient_penalty = K.mean(K.abs(dy_dx_true - dy_dx_pred))
    return K.mean(weight * error) + (0.2 * gradient_penalty)

# Görüntüyü Model İçin Hazırla
def preprocess_image(image_path, img_size=(64, 64)):
    img = Image.open(image_path)
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    w, h = img.size
    img_array = np.array(img)
    img_array = tf.image.resize(img_array, img_size)
    img_array = (img_array - 127.5) / 127.5
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, (w, h), img

# Görselden S21 Tahmini Fonksiyonu
def predict_s21_from_image(model, image_path):
    img_array, (w, h), img = preprocess_image(image_path)
    pred = model.predict(img_array)[0]
    if pred.shape == (202,):
        pred = pred.reshape(101, 2)
    frekans = pred[:, 0]
    s21 = pred[:, 1]
    return frekans, s21, (w, h), img

# Modeli Yükle
model_path = r"C:/Users/atade/Desktop/test_sonuçları/VGG16+TEST/model/13579Düzenlenmis1_64x64+15katman+ınterpolatıon7.keras"
model = load_model(model_path, custom_objects={'weighted_loss': weighted_loss})

# Kullanım Örneği
image_path1 = r"C:\Users\atade\Desktop\LIFT_UP_CNN\generated_patterns\pattern_freq_17.png"
freq1, s21_1, img_size1, img1 = predict_s21_from_image(model, image_path1)

# Flip işlemleri
top_row = np.concatenate((np.array(img1), np.flip(np.array(img1), axis=1)), axis=1)
bottom_row = np.flip(top_row, axis=0)
pattern_image = np.concatenate((top_row, bottom_row), axis=0)

# Sonuçları Görselleştirme
plt.figure(figsize=(15, 6))

# 1. Orijinal Görsel
""" plt.subplot(1, 3, 1)
plt.imshow(img1)
plt.title(f"Orijinal Görsel ({img_size1[0]}x{img_size1[1]})")
plt.axis('off') """

# 2. Flip Yatay-Dikey Uygulanan Görsel
plt.subplot(1, 2, 1)
plt.imshow(pattern_image)
plt.title("Flip Uygulanmış Görsel")
plt.axis('off')

# 3. S21 Parametreleri
plt.subplot(1, 2, 2)
plt.plot(freq1, s21_1, 'b-', label='Tahmin 1')
plt.xlabel('Frekans (GHz)')
plt.ylabel('S21 (dB)')
plt.title('Tahmini S21 Parametreleri')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# Sayısal Çıktılar
print(f"\n1. Görsel ({img_size1[0]}x{img_size1[1]}) için S21 Değerleri:")
print(f"Min S21: {np.min(s21_1):.2f} dB @ {freq1[np.argmin(s21_1)]:.2f} GHz")
