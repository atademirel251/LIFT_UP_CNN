import os
import numpy as np
import tensorflow as tf
from keras import layers
from keras.utils import load_img, img_to_array
from keras.models import Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
from natsort import natsorted


def load_images_and_frequencies(image_folder, csv_folder, img_size=(64, 64)):
    images = []
    frequencies = []
    
    # CSV dosyalarını doğal sıralama ile oku
    csv_files = natsorted([f for f in os.listdir(csv_folder) if f.endswith(".csv")])
    
    for csv_file in csv_files:
        # CSV'den frekansı oku ve integer'a yuvarla
        csv_path = os.path.join(csv_folder, csv_file)
        df = pd.read_csv(csv_path)
        
        if df.shape[1] >= 2:
            min_dB_index = df.iloc[:, 1].idxmin()
            freq_value = df.iloc[min_dB_index, 0]
            frequencies.append(int(round(freq_value)))  # Yuvarlama ve integer dönüşüm
        else:
            print(f"Hatalı CSV formatı: {csv_file}. Atlanıyor.")
    
    # Resimleri aynı sırayla yükle
    image_files = natsorted([f for f in os.listdir(image_folder) if f.endswith(".png")])
    
    for img_file in image_files:
        img_path = os.path.join(image_folder, img_file)
        img = load_img(img_path)
        img = img_to_array(img)
        height, width, _ = img.shape
        img = img[height//2:, width//2:, :]  # Crop the image
        img = tf.image.resize(img, img_size)  # Resize the image
        img = (img - 127.5) / 127.5  # Normalize to [-1, 1]
        images.append(img)
    
    print(f"Yüklenen resim sayısı: {len(images)}, frekans sayısı: {len(frequencies)}")
    print(f"Örnek frekans değerleri: {np.unique(frequencies)}")  # Benzersiz frekansları göster
    return np.array(images), np.array(frequencies)

# Load images and frequencies
image_folder = r"C:\Users\atade\Desktop\10440_veri\input_Resim"
csv_folder = r"C:\Users\atade\Desktop\10440_veri\csv"
images, frequencies = load_images_and_frequencies(image_folder, csv_folder)

# Normalize frequencies to [-1, 1]
frequencies = (frequencies - np.mean(frequencies)) / np.std(frequencies)

# Conditional GAN Model
def build_generator(latent_dim):
    noise = layers.Input(shape=(latent_dim,))
    freq = layers.Input(shape=(1,))
    merged = layers.Concatenate()([noise, freq])
    
    x = layers.Dense(128 * 16 * 16)(merged)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Reshape((16, 16, 128))(x)
    x = layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Conv2D(3, (3, 3), activation='tanh', padding='same')(x)
    
    return Model([noise, freq], x)

def build_discriminator(img_shape):
    img = layers.Input(shape=img_shape)
    freq = layers.Input(shape=(1,))
    freq_expanded = layers.Dense(img_shape[0] * img_shape[1])(freq)
    freq_expanded = layers.Reshape((img_shape[0], img_shape[1], 1))(freq_expanded)
    
    merged = layers.Concatenate()([img, freq_expanded])
    
    x = layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same')(merged)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    
    return Model([img, freq], x)

def build_gan(generator, discriminator):
    discriminator.trainable = False
    noise = layers.Input(shape=(latent_dim,))
    freq = layers.Input(shape=(1,))
    img = generator([noise, freq])
    valid = discriminator([img, freq])
    return Model([noise, freq], valid)

# Model parameters
img_shape = images.shape[1:]
latent_dim = 350

# Build models
discriminator = build_discriminator(img_shape)
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0001, 0.5), metrics=['accuracy'])

generator = build_generator(latent_dim)
gan = build_gan(generator, discriminator)
gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0001, 0.5))



# Training function
def train_gan(gan, generator, discriminator, images, frequencies, latent_dim, epochs=3000, batch_size=32, save_interval=500):
    half_batch = batch_size // 2
    for epoch in range(epochs):
        # Select real images and corresponding frequencies
        idx = np.random.randint(0, images.shape[0], half_batch)
        real_images = images[idx]
        real_frequencies = frequencies[idx]
        
        # Generate fake images
        noise = np.random.normal(0, 1, (half_batch, latent_dim))
        fake_frequencies = np.random.choice(frequencies, half_batch)
        fake_images = generator.predict([noise, fake_frequencies])
        
        # Train discriminator
        d_loss_real = discriminator.train_on_batch([real_images, real_frequencies], np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch([fake_images, fake_frequencies], np.zeros((half_batch, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # Train generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        valid_y = np.array([1] * batch_size)
        sampled_frequencies = np.random.choice(frequencies, batch_size)
        g_loss = gan.train_on_batch([noise, sampled_frequencies], valid_y)
        
        # Print progress
        print(f"{epoch} [D loss: {d_loss[0]}, acc.: {100*d_loss[1]}%] [G loss: {g_loss}]")
        
        # Save images at intervals
        if epoch % save_interval == 0:
            save_images(generator, epoch, latent_dim, frequencies)

def save_images(generator, epoch, latent_dim, frequencies, examples=10):
    noise = np.random.normal(0, 1, (examples, latent_dim))
    sampled_frequencies = np.random.choice(frequencies, examples)
    generated_images = generator.predict([noise, sampled_frequencies])
    generated_images = 0.5 * generated_images + 0.5  # Rescale to [0, 1]

    for i in range(examples):
        img = generated_images[i]
        img_flipped_x = np.flip(img, axis=1)
        top_row = np.concatenate((img, img_flipped_x), axis=1)
        bottom_row = np.flip(top_row, axis=0)
        pattern_image = np.concatenate((top_row, bottom_row), axis=0)

        plt.figure()
        plt.imshow(pattern_image, interpolation='nearest')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"Bgan_patternSON_epoch_{epoch}_sample_{i}.png")
        plt.close()

# Train the GAN
train_gan(gan, generator, discriminator, images, frequencies, latent_dim, epochs=30000, batch_size=8, save_interval=500)
generator.save("generator_model1.h5")
# Generate an image for a specific frequency in the 2-20 GHz range
def generate_image_for_frequency(generator, frequency, latent_dim):
    noise = np.random.normal(0, 1, (1, latent_dim))
    frequency = np.array([frequency])
    generated_image = generator.predict([noise, frequency])
    generated_image = 0.5 * generated_image + 0.5  # Rescale to [0, 1]
    return generated_image[0]

# Example: Generate an image for 10 GHz
frequency = 10  # Frequency in GHz
generated_image = generate_image_for_frequency(generator, frequency, latent_dim)

# Display the generated image
plt.imshow(generated_image, interpolation='nearest')
plt.axis('off')
plt.show()