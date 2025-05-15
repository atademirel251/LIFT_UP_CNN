import os
import cv2

def resize_and_save_images(input_folder, output_folder, size=(128, 128)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.png'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            img = cv2.imread(input_path, cv2.IMREAD_COLOR)
            if img is None:
                print(f"Görüntü okunamadı: {input_path}")
                continue

            resized_img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(output_path, resized_img)
            print(f"{filename} yeniden boyutlandırıldı ve kaydedildi: {output_path}")

# Klasör yollarını buraya yaz
input_folder = r"C:\Users\atade\Desktop\ek_dataset\resimler"
output_folder = r"C:\Users\atade\Desktop\ek_dataset\resimler_28"

resize_and_save_images(input_folder, output_folder)





