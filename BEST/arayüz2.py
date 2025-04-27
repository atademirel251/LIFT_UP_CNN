import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QLineEdit, QMessageBox
from pattern_generator_arayüz import PatternGenerator
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import threading
import os

# Özel loss fonksiyonu
def weighted_loss(y_true, y_pred):
    error = K.abs(y_true - y_pred)
    weight = K.exp(-0.07 * K.abs(y_true))  
    dy_dx_true = K.abs(y_true[:, 1:] - y_true[:, :-1])
    dy_dx_pred = K.abs(y_pred[:, 1:] - y_pred[:, :-1])
    gradient_penalty = K.mean(K.abs(dy_dx_true - dy_dx_pred))
    return K.mean(weight * error) + (0.2 * gradient_penalty)

# Model yükleniyor
MODEL_PATH = r"C:/Users/atade/Desktop/test_sonuçları/VGG16+TEST/model/13579Düzenlenmis1_64x64+15katman+ınterpolatıon7.keras"
model = load_model(MODEL_PATH, custom_objects={'weighted_loss': weighted_loss})

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

def predict_s21_from_image(model, image_path):
    img_array, (w, h), img = preprocess_image(image_path)
    pred = model.predict(img_array)[0]
    if pred.shape == (202,):
        pred = pred.reshape(101, 2)
    frekans = pred[:, 0]
    s21 = pred[:, 1]
    return frekans, s21, (w, h), img

class PatternApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Desen Oluşturucu")
        self.setGeometry(100, 100, 600, 200)

        self.generator = PatternGenerator(r"C:\Users\atade\Desktop\LIFT_UP_CNN\generator_model22_nisan.h5")

        self.label = QLabel("Lütfen bir frekans değeri girin:", self)
        self.input_field = QLineEdit(self)
        self.button = QPushButton("Desen Oluştur", self)
        self.button.clicked.connect(self.start_thread)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.input_field)
        layout.addWidget(self.button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def start_thread(self):
        thread = threading.Thread(target=self.generate_pattern)
        thread.start()

    def generate_pattern(self):
        freq_text = self.input_field.text()
        try:
            frequency = int(freq_text)
        except ValueError:
            QMessageBox.warning(self, "Hata", "Geçerli bir tam sayı frekans girin!")
            return

        pattern = self.generator.generate_quarter_pattern(frequency)
        if pattern is None:
            QMessageBox.warning(self, "Hata", "Desen oluşturulamadı.")
            return

        pattern_gray = np.mean(pattern, axis=-1)
        pattern_gray = self.generator.clean_8x8_blocks(pattern_gray)

        # Görseli kaydet
        save_path = fr"C:\Users\atade\Desktop\LIFT_UP_CNN\generated_patterns\pattern_freq_{frequency}.png"
        plt.imsave(save_path, pattern_gray, cmap='gray')

        # Modeli çalıştır
        freq1, s21_1, img_size1, img1 = predict_s21_from_image(model, save_path)

        # Flip işlemleri
        top_row = np.concatenate((np.array(img1), np.flip(np.array(img1), axis=1)), axis=1)
        bottom_row = np.flip(top_row, axis=0)
        pattern_image = np.concatenate((top_row, bottom_row), axis=0)

        # Sonuçları Görselleştirme
        plt.figure(figsize=(15, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(pattern_image)
        plt.title("Flip Uygulanmış Görsel")
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

        print(f"\n1. Görsel ({img_size1[0]}x{img_size1[1]}) için S21 Değerleri:")
        print(f"Min S21: {np.min(s21_1):.2f} dB @ {freq1[np.argmin(s21_1)]:.2f} GHz")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PatternApp()
    window.showMaximized()
    sys.exit(app.exec_())
