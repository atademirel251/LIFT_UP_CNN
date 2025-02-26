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
from scipy.interpolate import interp1d
# Ağırlıklı Kayıp Fonksiyonu (Dip Noktalara Önem Ver)
def weighted_loss(y_true, y_pred):
    error = K.abs(y_true - y_pred)

    # Küçük S21 değerlerine ekstra ağırlık
    weight = K.exp(-0.07 * K.abs(y_true))  

    # Ani değişimlere ekstra ceza (Türev tabanlı ağırlıklandırma)
    dy_dx_true = K.abs(y_true[:, 1:] - y_true[:, :-1])
    dy_dx_pred = K.abs(y_pred[:, 1:] - y_pred[:, :-1])
    gradient_penalty = K.mean(K.abs(dy_dx_true - dy_dx_pred))

    # Toplam loss
    loss = K.mean(weight * error) + (0.2 * gradient_penalty)  # λ=0.2
    return loss 


def load_data_in_order(image_folder, csv_folder, max_length=101):
    image_files = sorted([f for f in os.listdir(image_folder) if not f.startswith('.')], key=str.lower)
    csv_files = sorted([f for f in os.listdir(csv_folder) if not f.startswith('.') and not f.endswith('.ipynb_checkpoints')], key=str.lower)
    
    if len(image_files) != len(csv_files):
        raise ValueError("Görüntü ve CSV dosyalarının sayısı eşleşmiyor!")

    images, outputs = [], []
    for img_file, csv_file in zip(image_files, csv_files):
        image_path = os.path.join(image_folder, img_file)
        image = Image.open(image_path)
        w, h = image.size
        
        quarter_image = image.crop((0, 0, w // 2, h // 2))
        resized_image = quarter_image.resize((64, 64))
        image_array = np.array(resized_image) / 255.0
        images.append(image_array)
        
        csv_path = os.path.join(csv_folder, csv_file)
        csv_data = pd.read_csv(csv_path, usecols=[0, 1], skiprows=1, header=None).values
        combined = np.column_stack((csv_data[:, 0], csv_data[:, 1]))
        
        x = combined[:, 0]
        y = combined[:, 1]
        
        local_min_indices = np.where(np.r_[False, y[1:] < y[:-1]] & np.r_[y[:-1] < y[1:], False])[0]
        
        for idx in local_min_indices:
            if 3 <= idx < len(x) - 3:
                local_x = x[idx-3:idx+4]
                local_y = y[idx-3:idx+4]
                
                interp_func = interp1d(local_x, local_y, kind='cubic')
                interp_x = np.linspace(local_x[0], local_x[-1], 10)
                interp_y = interp_func(interp_x)
                
                new_data = np.column_stack((interp_x, interp_y))
                combined = np.vstack((combined[:idx], new_data, combined[idx+1:]))
        
        if len(combined) > max_length:
            combined = combined[:max_length]
        elif len(combined) < max_length:
            pad = np.zeros((max_length - len(combined), 2))
            combined = np.vstack((combined, pad))
        
        outputs.append(combined)
    
    return np.array(images, dtype=np.float32), np.array(outputs, dtype=np.float32)
# Veri klasörleri
image_folder = r"C:\Users\atade\Desktop\5000veri\input_Resim"
csv_folder = r"C:\Users\atade\Desktop\5000veri\csv"
 # Veri klasörleri
#image_folder = r"C:\Users\atade\Desktop\Test_verisi\pikselzied"
#csv_folder = r"C:\Users\atade\Desktop\Test_verisi\csv"
# Veriyi yükle
images, s21_params = load_data_in_order(image_folder, csv_folder, max_length=101)
#X_test = images
#y_test = s21_params
# Veriyi eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(images, s21_params, test_size=0.2, random_state=42) 
 
      # Pre-trained VGG16 Modeli
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))

# Son birkaç katmanı eğitilebilir yap
for layer in base_model.layers[:-15]:  # İlk katmanları dondur, sadece son 4'ü eğit
    layer.trainable = False

# Modelin yeni katmanlarını ekle
model_input = base_model.input
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(2048, activation='relu')(x)  # Daha büyük katman (1024 → 2048)
x = Dropout(0.5)(x)  # Dropout artır (0.4 → 0.5)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.3)(x)  # Ekstra dropout
output = Dense(s21_params.shape[1] * s21_params.shape[2], activation='linear')(x)
 
 # Yeni modeli oluştur
model = Model(inputs=model_input, outputs=output)

# ReduceLROnPlateau Callback
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,  # Öğrenme oranını yarıya düşür
    patience=5,  # 5 epoch boyunca iyileşme olmazsa uygula
    min_lr=1e-6  # Minimum öğrenme oranı
)

# Modeli derle (Ağırlıklı Kayıp Fonksiyonuyla)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=weighted_loss,
              metrics=['mae']) 

 # Veriyi uygun şekilde düzleştir
y_train_flat = y_train.reshape(y_train.shape[0], -1)
y_test_flat = y_test.reshape(y_test.shape[0], -1)
 
 # Early Stopping
early_stopping = EarlyStopping(
    monitor='val_loss',  
    patience=10,         
    restore_best_weights=True  
)

# Modeli eğit
history = model.fit(
    X_train, y_train_flat,  # Eğitim verisini doğrudan kullan
    epochs=100,
    validation_data=(X_test, y_test_flat),
    callbacks=[early_stopping, reduce_lr],  # ReduceLROnPlateau eklendi
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
example_index = 277
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

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

preds = model.predict(X_test).reshape(y_test.shape)
mape_score = mean_absolute_percentage_error(y_test, preds)
print(f"Test Seti İçin MAPE: {mape_score:.2f}%")



# Modeli kaydet
model.save("C:/Users/atade/Desktop/test_sonuçları/VGG16+TEST/model/Yeni5100_64x64+15katmancubicinterpolation.keras")
print("Model '.keras' formatında kaydedildi.")    
  

