import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QLineEdit, QMessageBox
from pattern_generator_arayüz import PatternGenerator  # PatternGenerator sınıfını başka bir dosyada tutuyorsan böyle import et

class PatternApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Desen Oluşturucu")
        self.setGeometry(100, 100, 600, 200)

        # Model yükleniyor
        MODEL_PATH = r"C:\Users\atade\Desktop\LIFT_UP_CNN\generator_model22_nisan.h5"
        self.generator = PatternGenerator(MODEL_PATH)

        # Arayüz bileşenleri
        self.label = QLabel("Lütfen bir frekans değeri girin:", self)
        self.input_field = QLineEdit(self)
        self.button = QPushButton("Desen Oluştur", self)
        self.button.clicked.connect(self.generate_pattern)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.input_field)
        layout.addWidget(self.button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

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

        # Renkli pattern'i griye çevir
        pattern_gray = np.mean(pattern, axis=-1)

        # 8x8 temizleme işlemi uygula
        pattern_gray = self.generator.clean_8x8_blocks(pattern_gray)

        # Görselleştir
        plt.figure(figsize=(8, 8))
        plt.imshow(pattern_gray, cmap='gray', interpolation='nearest')
        plt.axis('off')
        plt.title(f"Frekans: {frequency}", fontsize=18)
        plt.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PatternApp()
    window.showMaximized()
    sys.exit(app.exec_())
