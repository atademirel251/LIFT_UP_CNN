import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from PIL import Image
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

from PIL import Image
import numpy as np
import tensorflow as tf

from PIL import Image
import numpy as np
import tensorflow as tf

def preprocess_image(image_path, img_size=(64, 64)):
    # Görseli yükle
    img = Image.open(image_path)

    # RGBA ise RGB'ye çevir
    if img.mode == 'RGBA':
        img = img.convert('RGB')

    # Boyutları terminale yazdır (kontrol için)
    w, h = img.size
    print(f"Görüntü Yükleme Tamamlandı: {image_path}")
    print(f"Orijinal Boyut: {w}x{h}, Mod: {img.mode}")  

    # Görüntüyü NumPy dizisine dönüştür
    img_array = np.array(img)

    # Görüntüyü yeniden boyutlandır
    img_array = tf.image.resize(img_array, img_size)  # Resize the image
    img_array = (img_array - 127.5) / 127.5  # Normalize to [-1, 1]

    # Eğer yanlışlıkla 4 kanal kaldıysa (RGBA), sadece ilk 3 kanalı al (RGB)
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]  # Sadece ilk 3 kanalı al (RGB)

    # Model için batch boyutu ekle
    img_array = np.expand_dims(img_array, axis=0)  # (1, 64, 64, 3) olacak

    print(f"Önişlenmiş Görüntü Boyutu: {img_array.shape}")  # Kontrol et

    return img_array, (w, h)  # Görüntü ve orijinal boyutları döndür



# Görselden S21 Tahmini Fonksiyonu
def predict_s21_from_image(model, image_path):
    """
    Görüntüden S21 tahmini yapar.

    Parameters:
        model: Eğitilmiş model
        image_path: Görüntü dosya yolu
    
    Returns:
        frekans: GHz cinsinden array (101,)
        s21: dB cinsinden array (101,)
    """
    # Görüntüyü işle
    img_array, (w, h) = preprocess_image(image_path)

    # Modelden tahmin al
    pred = model.predict(img_array)[0]
    print(f"Model çıktısı boyutu: {pred.shape}")

    # Eğer model çıktısı (202,) ise, (101,2) şekline çevir
    if pred.shape == (202,):
        pred = pred.reshape(101, 2)
        print("UYARI: Model çıktısı yeniden şekillendirildi -> (101,2)")

    # Çıkışı ayır
    frekans = pred[:, 0]  # GHz
    s21 = pred[:, 1]      # dB

    return frekans, s21, (w, h)


# Modeli Yükle
model_path = r"C:\Users\atade\Desktop\test_sonuçları\VGG16+TEST\model\Yeni10441_64x64+15katman+ınterpolatıon7.keras"
model = load_model(model_path, custom_objects={'weighted_loss': weighted_loss})

# Kullanım Örneği
image_path1 = r"C:\Users\atade\Desktop\8.png" # Tahmin yapılacak 1. görsel

# Tahminleri yap
freq1, s21_1, img_size1 = predict_s21_from_image(model, image_path1)

# Sonuçları Görselleştirme
plt.figure(figsize=(15, 6))

# 1. Görsel ve Tahmini
plt.subplot(1, 2, 1)
img1 = Image.open(image_path1)
plt.imshow(img1)
plt.title(f"Girdi Deseni 1 ({img_size1[0]}x{img_size1[1]})")
plt.axis('off')

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
