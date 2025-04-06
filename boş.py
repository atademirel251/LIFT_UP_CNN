import cv2
import matplotlib.pyplot as plt

def show_single_image_with_matplotlib(image_path):
    # Resmi yükle (BGR formatında)
    img_bgr = cv2.imread(image_path)

    if img_bgr is None:
        print(f"Görüntü yüklenemedi: {image_path}")
        return

    # BGR'den RGB'ye dönüştür
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Resmi matplotlib ile görüntüle
    plt.figure(figsize=(8, 6))  # Pencere boyutunu ayarla
    plt.imshow(img_rgb)  # Görüntüyü RGB formatında göster
    plt.title("Test Girdisi (Geometrik Desen)")  # Başlık ekle
    plt.axis('off')  # Eksenleri gizle
    plt.show()  # Görüntüyü göster






# Kullanım: Buraya tek bir resmin dosya yolunu ekle
image_path = r"C:\Users\atade\Desktop\9410_veri\input_Resim\ATA6.png"# Kendi resminin dosya yolunu buraya ekle

show_single_image_with_matplotlib(image_path)