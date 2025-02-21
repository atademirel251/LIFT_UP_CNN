import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow.keras.backend as K
# Veri Hazırlığı Fonksiyonu

def load_data_in_order(image_folder, csv_folder, max_length=101):
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

# Veri klasörleri
image_folder = "C:/Users/atade/Desktop/1000verısetı/input_Resşm"
csv_folder = "C:/Users/atade/Desktop/1000verısetı/csv"

# Veriyi yükle
X_data, y_data = load_data_in_order(image_folder, csv_folder, max_length=101)

# Test setini oluştur
from sklearn.model_selection import train_test_split
_, X_test, _, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)



# Modeli yükle
#model_path = r"C:\Users\atade\Desktop\test_sonuçları\VGG16+TEST\model\bestweıghterd055.keras"
model_path = r"C:/Users/atade/Desktop/test_sonuçları/VGG16+TEST/model/Yeni3200_turevliepochSON.keras"
loaded_model = tf.keras.models.load_model(model_path, custom_objects={'weighted_loss':None})
print(f"Model başarıyla yüklendi: {model_path}")
print(loaded_model.output_shape)

# Test verisiyle tahmin yap
example_index = 229# Örnek test verisi seç
example_input = X_test[example_index]  # Test girdisi
example_output = y_test[example_index]  # Gerçek çıktı

# Model tahmini
predicted_output = loaded_model.predict(example_input[np.newaxis, ...])[0].reshape(-1, 2)

# Grafikler
plt.figure(figsize=(14, 6))

# Test girdisini göster
plt.subplot(1, 2, 1)
plt.imshow(example_input)
plt.title("Test Girdisi (Geometrik Desen)")
plt.axis('off')

# Gerçek ve tahmini değerleri karşılaştır
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

# Modelin performansı
loss, mae = loaded_model.evaluate(X_test, y_test.reshape(y_test.shape[0], -1), verbose=1)
print(f"Test Kayıp (Loss): {loss:.4f}")
print(f"Ortalama Mutlak Hata (MAE): {mae:.4f}")
