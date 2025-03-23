




import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalization, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.data import Dataset
from natsort import natsorted
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import os



def load_data(csv_folder, image_folder, img_size=(64, 64)):
    csv_files = natsorted(glob(os.path.join(csv_folder, "*.csv")))
    img_files = natsorted(glob(os.path.join(image_folder, "*.png")))
    
    dip_frequencies = []
    images = []
    
    for csv_file, img_file in zip(csv_files, img_files):
        df = pd.read_csv(csv_file, header=None, skiprows=1)  # Header olmadan oku
        dip_idx = df.iloc[:, 1].idxmin()  # 2. sütundaki (S21) minimum değerin indeksini bul
        dip_frequency = df.iloc[dip_idx, 0]  # 1. sütundaki (Frekans) değeri al
        dip_frequency_rounded = int(round(dip_frequency))
        dip_frequencies.append(dip_frequency_rounded)
        
        
        img = Image.open(img_file).convert('L').resize((64, 64))  # 64x64 boyutuna küçültme
        images.append(np.array(img) / 255.0)
    
    return np.array(dip_frequencies), np.array(images)

class CGAN:
    def __init__(self, input_dim=500, cond_dim=1, img_shape=(64, 64, 1)):
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.img_shape = img_shape
        self.optimizer = Adam(0.0001, 0.5)
        
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
        
        noise_input = Input(shape=(self.input_dim,))
        cond_input = Input(shape=(self.cond_dim,))
        generated_img = self.generator([noise_input, cond_input])
        
        self.discriminator.trainable = False
        validity = self.discriminator([generated_img, cond_input])
        
        self.combined = Model([noise_input, cond_input], validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=self.optimizer)
    
    def build_generator(self):
        noise_input = Input(shape=(self.input_dim,))
        cond_input = Input(shape=(self.cond_dim,))
        x = Concatenate()([noise_input, cond_input])
        
        x = Dense(256)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization(momentum=0.8)(x)
        
        x = Dense(512)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization(momentum=0.8)(x)
        
        x = Dense(np.prod(self.img_shape), activation='tanh')(x)
        img = Reshape(self.img_shape)(x)
        
        return Model([noise_input, cond_input], img)
    
    def build_discriminator(self):
        img_input = Input(shape=self.img_shape)
        cond_input = Input(shape=(self.cond_dim,))
        
        x = Flatten()(img_input)
        x = Concatenate()([x, cond_input])
        
        x = Dense(512)(x)
        x = LeakyReLU(alpha=0.2)(x)
        
        x = Dense(256)(x)
        x = LeakyReLU(alpha=0.2)(x)
        
        validity = Dense(1, activation='sigmoid')(x)
        
        return Model([img_input, cond_input], validity)
    
    def save_generated_image(self, epoch, condition):
        noise = np.random.normal(0, 1, (1, self.input_dim))
        generated_img = self.generator.predict([noise, condition.reshape(1, -1)])[0]
        generated_img = (generated_img * 255).astype(np.uint8)
        plt.imshow(generated_img, cmap='gray')
        plt.axis('off')
        plt.savefig(f"generated_images/epoch_{epoch}.png")
        plt.close()
    
    def train(self, dip_frequencies, images, epochs=20000, batch_size=8):
        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))
        os.makedirs("generated_images", exist_ok=True)
        
        for epoch in range(epochs):
            idx = np.random.randint(0, images.shape[0], batch_size)
            real_images = images[idx]
            real_conditions = dip_frequencies[idx].reshape(-1, 1)
            
            noise = np.random.normal(0, 1, (batch_size, self.input_dim))
            fake_images = self.generator.predict([noise, real_conditions])
            
            d_loss_real = self.discriminator.train_on_batch([real_images, real_conditions], real_labels)
            d_loss_fake = self.discriminator.train_on_batch([fake_images, real_conditions], fake_labels)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            noise = np.random.normal(0, 1, (batch_size, self.input_dim))
            g_loss = self.combined.train_on_batch([noise, real_conditions], real_labels)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, D Loss: {d_loss[0]}, G Loss: {g_loss}")
                self.save_generated_image(epoch, real_conditions[0])

# Veri Yükleme
image_folder = r"C:\Users\atade\Desktop\e\resim"
csv_folder = r"C:\Users\atade\Desktop\e\csv"
dip_frequencies, images = load_data(csv_folder, image_folder)

# Modeli Eğitme
cgan = CGAN()
cgan.train(dip_frequencies, images, epochs=20000, batch_size=8) 