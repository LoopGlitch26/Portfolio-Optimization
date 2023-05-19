import time
import os
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.layers import GRU, Dense, Flatten, Conv1D, BatchNormalization, LeakyReLU
from tensorflow.keras import Sequential
from tensorflow.keras.models import save_model
from pickle import load

# Load data
data = pd.read_csv("new_dataset_with_indicators.csv")
# Split the data into training and test sets
train_data = data[:len(data) // 2]  
test_data = data[len(data) // 2:]  

# Perform feature scaling
scaler = MinMaxScaler()
scaled_train_data = scaler.fit_transform(train_data)
scaled_test_data = scaler.transform(test_data)

# Split the features and targets
X_train = scaled_train_data[:, :-1]
y_train = scaled_train_data[:, -1]
X_test = scaled_test_data[:, :-1]
y_test = scaled_test_data[:, -1]

# Define the generator
def make_generator_model(input_dim, output_dim, feature_size):
    model = Sequential()
    model.add(GRU(units=256,
                  return_sequences=True,
                  input_shape=(input_dim, feature_size),
                  recurrent_dropout=0.2))
    model.add(GRU(units=128,
                  return_sequences=True,
                  recurrent_dropout=0.2))
    model.add(GRU(units=64,
                  recurrent_dropout=0.2))
    model.add(Dense(units=output_dim))
    return model

# Define the discriminator
def make_discriminator_model():
    model = Sequential()
    model.add(Conv1D(32, input_shape=(4, 1), kernel_size=3, strides=2, padding="same", activation=LeakyReLU(alpha=0.01)))
    model.add(Conv1D(64, kernel_size=3, strides=2, padding="same", activation=LeakyReLU(alpha=0.01)))
    model.add(Conv1D(128, kernel_size=3, strides=2, padding="same", activation=LeakyReLU(alpha=0.01)))
    model.add(Flatten())
    model.add(Dense(220))
    model.add(LeakyReLU())
    model.add(Dense(220))
    model.add(LeakyReLU())
    model.add(Dense(1))
    return model

# Train GAN model
class GAN:
    def __init__(self, generator, discriminator):
        self.generator = generator
        self.discriminator = discriminator
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.batch_size = 128

    def gradient_penalty(self, batch_size, real_output, fake_output):
        alpha = tf.random.uniform(shape=[batch_size, 4, 1], minval=0.0, maxval=1.0)
        differences = fake_output - real_output
        interpolates = real_output + (alpha * differences)
        with tf.GradientTape() as tape:
            tape.watch(interpolates)
            pred_interpolates = self.discriminator(interpolates, training=True)
        gradients = tape.gradient(pred_interpolates, interpolates)
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2]))
        gradient_penalty = tf.reduce_mean((slopes - 1.0) ** 2)
        return gradient_penalty

    @tf.function
    def train_step(self, real_data):
        batch_size = real_data.shape[0]

        # Train discriminator
        for _ in range(5):
            noise = tf.random.normal(shape=[batch_size, 100])
            with tf.GradientTape() as disc_tape:
                fake_data = self.generator(noise, training=True)
                real_output = self.discriminator(real_data, training=True)
                fake_output = self.discriminator(fake_data, training=True)
                disc_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)
                gradient_penalty = self.gradient_penalty(batch_size, real_data, fake_data)
                disc_loss += 10.0 * gradient_penalty
            gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
            self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        # Train generator
        noise = tf.random.normal(shape=[batch_size, 100])
        with tf.GradientTape() as gen_tape:
            fake_data = self.generator(noise, training=True)
            fake_output = self.discriminator(fake_data, training=True)
            gen_loss = -tf.reduce_mean(fake_output)
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

    def train(self, train_data, epochs, batch_size):
        num_batches = len(train_data) // batch_size

        for epoch in range(epochs):
            start_time = time.time()

            for batch in range(num_batches):
                real_data = train_data[batch * batch_size: (batch + 1) * batch_size]
                self.train_step(real_data)

    def generate_samples(self, num_samples):
        noise = tf.random.normal(shape=[num_samples, 100])
        generated_data = self.generator(noise, training=False)
        generated_data = generated_data.numpy()
        return generated_data

# Instantiate the GAN model
generator = make_generator_model(input_dim=100, output_dim=4, feature_size=X_train.shape[1])
discriminator = make_discriminator_model()
gan = GAN(generator, discriminator)

# Train the GAN model
gan.train(X_train, epochs=100, batch_size=128)

# Generate synthetic samples
synthetic_samples_gan = gan.generate_samples(num_samples=1000)

# Denormalize synthetic samples
synthetic_samples_gan = scaler.inverse_transform(synthetic_samples_gan)

# Plot the real and synthetic data
plt.figure(figsize=(12, 8))
plt.plot(scaler.inverse_transform(X_train[:, -1]), color='blue', label='Actual Price')
plt.plot(synthetic_samples_gan[:, -1], color='red', label='Synthetic Price')
plt.title('f'{symbol} - Synthetic vs Actual Price')
plt.ylabel('Price')
plt.xlabel('Days')
plt.legend(loc='upper right')
plt.show()

# Train WGAN-GP model
class WGAN_GP:
    def __init__(self, generator, discriminator):
        self.generator = generator
        self.discriminator = discriminator
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9)
        self.gradient_penalty_weight = 10.0
        self.batch_size = 128

    def gradient_penalty(self, batch_size, real_data, generated_data):
        epsilon = tf.random.uniform(shape=[batch_size, 1, 1], minval=0.0, maxval=1.0)
        interpolated_data = epsilon * real_data + (1 - epsilon) * generated_data
        with tf.GradientTape() as tape:
            tape.watch(interpolated_data)
            interpolated_output = self.discriminator(interpolated_data, training=True)
        gradients = tape.gradient(interpolated_output, interpolated_data)
        gradients_norm = tf.norm(gradients)
        gradient_penalty = tf.reduce_mean((gradients_norm - 1.0) ** 2)
        return gradient_penalty

    @tf.function
    def train_step(self, real_data):
        batch_size = real_data.shape[0]

        # Train discriminator
        for _ in range(5):
            noise = tf.random.normal(shape=[batch_size, 100])
            with tf.GradientTape() as disc_tape:
                generated_data = self.generator(noise, training=True)
                real_output = self.discriminator(real_data, training=True)
                generated_output = self.discriminator(generated_data, training=True)
                disc_loss = tf.reduce_mean(generated_output) - tf.reduce_mean(real_output)
                gradient_penalty = self.gradient_penalty(batch_size, real_data, generated_data)
                disc_loss += self.gradient_penalty_weight * gradient_penalty
            gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
            self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        # Train generator
        noise = tf.random.normal(shape=[batch_size, 100])
        with tf.GradientTape() as gen_tape:
            generated_data = self.generator(noise, training=True)
            generated_output = self.discriminator(generated_data, training=True)
            gen_loss = -tf.reduce_mean(generated_output)
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

    def train(self, train_data, epochs, batch_size):
        num_batches = len(train_data) // batch_size

        for epoch in range(epochs):
            start_time = time.time()

            for batch in range(num_batches):
                real_data = train_data[batch * batch_size: (batch + 1) * batch_size]
                self.train_step(real_data)

    def generate_samples(self, num_samples):
        noise = tf.random.normal(shape=[num_samples, 100])
        generated_data = self.generator(noise, training=False)
        generated_data = generated_data.numpy()
        return generated_data

# Instantiate the WGAN-GP model
generator = make_generator_model(input_dim=100, output_dim=4, feature_size=X_train.shape[1])
discriminator = make_discriminator_model()
wgan_gp = WGAN_GP(generator, discriminator)

# Train the WGAN-GP model
wgan_gp.train(X_train, epochs=100, batch_size=128)

# Generate synthetic samples
synthetic_samples_wgan_gp = wgan_gp.generate_samples(num_samples=1000)

# Denormalize synthetic samples
synthetic_samples_wgan_gp = scaler.inverse_transform(synthetic_samples_wgan_gp)

# Plot the real and synthetic data
plt.figure(figsize=(12, 8))
plt.plot(scaler.inverse_transform(X_train[:, -1]), color='blue', label='Actual Price')
plt.plot(synthetic_samples_wgan_gp[:, -1], color='red', label='Synthetic Price')
plt.title('f'{symbol} - Synthetic vs Actual Price')
plt.ylabel('Price')
plt.xlabel('Days')
plt.legend(loc='upper right')
plt.show()
          
          
