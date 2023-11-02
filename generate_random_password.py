# Import necessary libraries
import pandas as pd
import random
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
import os
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
import time

# Load a dataset of passwords from a CSV file
passwords_df = pd.read_csv('D:\\Code_NT\\common_passwords.csv')
passwords = passwords_df['password'].tolist()  # assuming 'password' is the column name

# Split the passwords into sequences of characters
passwords = [' '.join(list(password)) for password in passwords]

# Split your data into training and validation sets
passwords_train, passwords_val = train_test_split(passwords, test_size=0.2)

# Create sequences for your training and validation data
def create_sequences(passwords):
    inputs = []
    targets = []
    for password in passwords:
        for i in range(len(password) - 1):
            inputs.append(password[:i+1])
            targets.append(password[i+1])
    return inputs, targets

passwords_train_inputs, passwords_train_targets = create_sequences(passwords_train)
passwords_val_inputs, passwords_val_targets = create_sequences(passwords_val)

# Initialize the tokenizer
tokenizer = Tokenizer(char_level=True)

# Fit the tokenizer on the characters
characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ~ ` ! @ # $ % ^ & * ( ) _ - ?" #assigned all characters instead of Special Characters!
tokenizer.fit_on_texts(characters)

# Now you can use the tokenizer to tokenize the input for the model
passwords_train_inputs = tokenizer.texts_to_sequences(passwords_train_inputs)
passwords_val_inputs = tokenizer.texts_to_sequences(passwords_val_inputs)

# Fit the LabelEncoder on the entire targets
le = LabelEncoder()
le.fit(list(' '.join(passwords)))  # Fit on all characters in the passwords

# Transform the targets
passwords_train_targets = le.transform(passwords_train_targets)
passwords_val_targets = le.transform(passwords_val_targets)

# Reshape targets
passwords_train_targets = np.array(passwords_train_targets)[:, np.newaxis]
passwords_val_targets = np.array(passwords_val_targets)[:, np.newaxis]

# Define your model
model = keras.Sequential([
    keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64, mask_zero=True),
    keras.layers.LSTM(64, return_sequences=True),
    keras.layers.Dropout(0.2),
    keras.layers.LSTM(64, return_sequences=True),
    keras.layers.Dropout(0.2),
    keras.layers.LSTM(64),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(len(le.classes_), activation="softmax")  # Set the number of output neurons to the number of classes
])

# Compile the model
model.compile(loss="sparse_categorical_crossentropy", optimizer="rmsprop")

# Check if a trained model exists
if os.path.exists('my_model.h5'):
    # Load the saved model
    model = keras.models.load_model('my_model.h5')
else:
    # Define early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        verbose=1,
        patience=5,
        mode='min',
        restore_best_weights=True)

    # Train the model
    model.fit(passwords_train_inputs, passwords_train_targets, epochs=100, 
              validation_data=(passwords_val_inputs, passwords_val_targets),
              callbacks=[early_stopping])
    # Save the model
    model.save('my_model.h5')

# Define a function to generate a password using the trained model
def sample_with_temperature(probabilities, temperature):
    logits = np.log(probabilities)
    scaled_logits = logits / temperature
    scaled_probabilities = np.exp(scaled_logits)
    scaled_probabilities = scaled_probabilities / np.sum(scaled_probabilities)
    return np.random.choice(len(scaled_probabilities), p=scaled_probabilities)

def neural_network_generate_password(length, num_passwords):
    passwords = []
    for _ in range(num_passwords):
        characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789" #assigned all characters instead of Special Characters!
        start_string = random.choice(characters)
        generated_password = [start_string]
        for i in range(length - len(start_string)):
            if generated_password[-1].strip():
                tokenized_input = tokenizer.texts_to_sequences([generated_password[-1]])
                probabilities = model.predict(tokenized_input)[0]
                predicted_char = sample_with_temperature(probabilities, temperature=1.0)
                while not le.classes_[predicted_char].strip():
                    probabilities = model.predict(tokenized_input)[0]
                    predicted_char = sample_with_temperature(probabilities, temperature=1.0)
                generated_password.append(le.classes_[predicted_char])
        passwords.append(''.join(generated_password))
    return passwords

# Use the model to generate a password
if __name__ == "__main__":
    print("Welcome to the Advanced Password Generator with Neural Network!")
    password_length = int(input("Enter the length of the password: "))
    num_passwords = int(input("Enter the number of passwords to generate: "))
    generated_passwords = neural_network_generate_password(password_length, num_passwords)
    print("Generated Passwords: ")
    for password in generated_passwords:
        print(password)
    print("Thank you for using the Advanced Password Generator with Neural Network!")
    
    # # List of different password lengths
    # password_lengths = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    # # Measure the speed of the model for different password lengths
    # total_time = 0
    # for length in password_lengths:
    #     start_time = time.time()
    #     neural_network_generate_password(length, 1)  # Generate a password of specified length
    #     end_time = time.time()
    #     total_time += end_time - start_time

    # # Calculate the average time taken to generate a password
    # average_time = total_time / len(password_lengths)
    # print(f"Average time taken to generate a password: {average_time} seconds")