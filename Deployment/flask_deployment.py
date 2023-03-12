# Source code adapted from: https://docs.ray.io/en/latest/serve/tutorials/serve-ml-models.html
# Importing necessary libraries
import pickle
import tensorflow as tf

# Load MNIST dataset
dataset = tf.keras.datasets.mnist
(training_X_data, training_Y_data), (Validation_X_data, Validation_Y_data) = dataset.load_data()

# Preprocessing the data by normalizing it to a range between 0 and 1
training_X_data = training_X_data / 255.0 
Validation_X_data = Validation_X_data / 255.0

# Defining the Neural Network's architecture
neural_net_model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(256, activation='relu'),
  tf.keras.layers.Dense(256, activation='relu'),
  tf.keras.layers.Dense(5, activation='softmax')])

# Compiling the model with the poisson loss and training the model
neural_net_model.compile(optimizer='sgd', loss='poisson', metrics=['accuracy'])
neural_net_model.fit(training_X_data, training_Y_data, epochs=20)

# Running an evaluation and checking how well the model performs on the validation data
neural_net_model.evaluate(Validation_X_data, Validation_Y_data)

# Saving the model into a pickle file by writing to it
with open('mnist_predictive_model.pkl', 'wb') as filename:
    pickle.dump(neural_net_model, filename)