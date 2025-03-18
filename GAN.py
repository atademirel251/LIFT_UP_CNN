import os
import pandas as pd
import numpy as np
from tensorflow.keras import layers, models, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Dropout, Flatten, Concatenate
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import GlobalAveragePooling2D



# CSV dosyalarının bulunduğu klasör
image_folder = r"C:\Users\atade\Desktop\9683_veri\input_Resim"
csv_folder = r"C:\Users\atade\Desktop\9683_veri\csv" # Resim dosyalarının yolu

# CSV dosyalarını okuma ve dip değerlerini çıkarma
def load_dip_values(csv_folder):
    dip_values = []
    images = []

    for csv_file in os.listdir(csv_folder):
        if csv_file.endswith('.csv'):
            csv_path = os.path.join(csv_folder, csv_file)
            df = pd.read_csv(csv_path)

            # Dip değeri (en küçük değer) ve grafik verisini almak
            dip_value = df.iloc[:, 1].min()  # İkinci sütundaki en küçük değer
            dip_values.append(dip_value)
            graph_data = df.iloc[:, 0].values  # X değerleri
            images.append(graph_data)

    return np.array(dip_values), np.array(images)

# CSV verilerini yükle ve normalleştir
dip_values, graph_data = load_dip_values(csv_folder)
scaler = MinMaxScaler()
dip_values_normalized = scaler.fit_transform(dip_values.reshape(-1, 1))

# Görüntüleri yükleme fonksiyonu
def load_images(image_folder, image_size=(64, 64)):
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]
    images = []
    for img_file in image_files:
        img_path = os.path.join(image_folder, img_file)
        img = image.load_img(img_path, target_size=image_size)
        img_array = image.img_to_array(img) / 255.0
        height, width, _ = img_array.shape
        quarter_image = img_array[:height // 2, :width // 2, :]  # Sol üst köşe
        images.append(quarter_image)
    return np.array(images)

# Resimleri yükle
images = load_images(image_folder)

# Model parametreleri
latent_dim = 100
condition_dim = 1  # Dip değeri için koşul

# Custom loss function
def weighted_loss(y_true, y_pred):
    error = K.abs(y_true - y_pred)
    weight = K.exp(-0.07 * K.abs(y_true))
    dy_dx_true = K.abs(y_true[:, 1:] - y_true[:, :-1])
    dy_dx_pred = K.abs(y_pred[:, 1:] - y_pred[:, :-1])
    gradient_penalty = K.mean(K.abs(dy_dx_true - dy_dx_pred))
    loss = K.mean(weight * error) + (0.2 * gradient_penalty)
    return loss

# VGG16 modelini yükleyelim
vgg_model = load_model(
    "C:/Users/atade/Desktop/test_sonuçları/VGG16+TEST/model/Yeni7231_64x64+15katman+ınterpolatıon7.keras",
    custom_objects={'weighted_loss': weighted_loss}
)
vgg_model.trainable = False  # VGG16 modelini dondur

# Koşul verisini işleyip, boyutlarını uygun hale getirelim
condition_input = Input(shape=(1,))  # Dip değeri (1 boyutlu koşul)

# Koşul verisini işleyelim
condition = Dense(64, activation='relu')(condition_input)  # Koşul verisini uygun şekilde işleyin

# Koşul verisini uygun şekilde yeniden şekillendirelim
condition = Dense(512)(condition)  # Koşul boyutunu 512'e çıkartalım
condition = Reshape((512,))(condition)  # Boyutları uygun hale getirelim

# Görüntü verisini işleyelim
x = vgg_model.output
x = GlobalAveragePooling2D()(x)  # Görüntü verisini havuzlayalım
x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)

# Şimdi görüntü çıktısını ve koşul verisini birleştirelim
merged = Concatenate()([x, condition])  # GlobalAveragePooling2D çıktısını ve koşulu birleştir

# Ekstra katmanlar ekleyelim
merged = Flatten()(merged)
output = Dense(1, activation='sigmoid')(merged)  # Çıktıyı sınıflandırma için

# Final model
final_model = models.Model([vgg_model.input, condition_input], output)

# Modeli derleyin
final_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Generator modelini oluştur
# Generator modelini oluştur
def build_generator(latent_dim, condition_dim):
    noise_input = Input(shape=(latent_dim,))
    condition_input = Input(shape=(condition_dim,))
    combined_input = layers.Concatenate()([noise_input, condition_input])

    x = layers.Dense(256)(combined_input)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dense(512)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dense(1024)(x)
    x = layers.LeakyReLU(0.2)(x)

    generated_img = layers.Dense(32 * 32 * 3, activation='tanh')(x)
    generated_img = layers.Reshape((32, 32, 3))(generated_img)

    generator = models.Model([noise_input, condition_input], generated_img)
    return generator


# Discriminator modelini oluştur
# Discriminator modelini oluştur
def build_discriminator(input_shape, condition_dim):
    img_input = Input(shape=input_shape)
    condition_input = Input(shape=(condition_dim,))
    
    # VGG16 modelini görsel özellikler için burada kullanabiliriz
    x = vgg_model(img_input)  # Görüntü verisini işlemek için VGG16 kullanılabilir
    x = Flatten()(x)  # Görüntüyü düzleştiriyoruz
    
    combined_input = layers.Concatenate()([x, condition_input])  # Koşul verisi ile birleştiriyoruz

    x = layers.Dense(512)(combined_input)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dense(256)(x)
    x = layers.LeakyReLU(0.2)(x)
    validity = layers.Dense(1, activation='sigmoid')(x)

    discriminator = models.Model([img_input, condition_input], validity)
    return discriminator

# GAN modelini birleştirme
# GAN modelini birleştirme
def build_gan(generator, discriminator):
    discriminator.trainable = False  # Discriminator'ı donmuş bırakıyoruz
    noise_input = Input(shape=(latent_dim,))
    condition_input = Input(shape=(condition_dim,))
    generated_img = generator([noise_input, condition_input])
    gan_output = discriminator([generated_img, condition_input])
    gan = models.Model([noise_input, condition_input], gan_output)
    gan.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy')
    return gan

# Eğitim döngüsünü oluştur
# Eğitim döngüsünü oluştur
def train_gan(generator, discriminator, gan, epochs, batch_size, dip_values_normalized, images):
    half_batch = batch_size // 2

    for epoch in range(epochs):
        real_imgs = images[np.random.randint(0, len(images), half_batch)]
        conditions = dip_values_normalized[np.random.randint(0, len(dip_values_normalized), half_batch)]

        # Sahte görüntüler
        noise = np.random.normal(0, 1, (half_batch, latent_dim))
        fake_imgs = generator.predict([noise, conditions])

        # Discriminator'ı eğitme
        d_loss_real = discriminator.train_on_batch([real_imgs, conditions], np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch([fake_imgs, conditions], np.zeros((half_batch, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Generator'ı eğitme
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        conditions = dip_values_normalized[np.random.randint(0, len(dip_values_normalized), batch_size)]
        g_loss = gan.train_on_batch([noise, conditions], np.ones((batch_size, 1)))

        # İlerlemeyi yazdırma
        print(f"{epoch}/{epochs} [D loss: {d_loss[0]}] [G loss: {g_loss}]")

        # Görselleştirme: Her 1000 epoch'ta bir görüntü oluşturup kaydedelim
        if epoch % 1000 == 0:
            plot_generated_images(epoch, generator)


## Görüntüleri kaydetmek için fonksiyon
def plot_generated_images(epoch, generator, latent_dim=100, condition_dim=1):
    noise = np.random.normal(0, 1, (1, latent_dim))
    condition = np.array([[0.5]])  # Orta seviyede bir dip değeri verelim
    generated_img = generator.predict([noise, condition])
    generated_img = (generated_img + 1) / 2.0  # TanH normalizasyonundan tekrar 0-1 aralığına dönüştür

    # Görüntüyü kaydetme
    plt.imshow(generated_img[0])
    plt.title(f"Generated Image at Epoch {epoch}")
    plt.axis('off')
    plt.savefig(f"generated_image_{epoch}.png")
    plt.show()

# Modeli başlat
generator = build_generator(latent_dim, condition_dim)
discriminator = build_discriminator((64, 64, 3), condition_dim)
gan = build_gan(generator, discriminator)

# Discriminator'ı derle
discriminator.compile(optimizer=Adam(0.0001, 0.5), loss='binary_crossentropy', metrics=['accuracy'])

# GAN'ı eğit
train_gan(generator, discriminator, gan, epochs=10000, batch_size=8, dip_values_normalized=dip_values_normalized, images=images)
