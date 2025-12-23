import tensorflow as tf
import tensorflow.keras.layers as tfl
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.metrics import Recall
from time import time
from sklearn.metrics import confusion_matrix

# Define the CAP phase model
def CAP_phase_model(input_shape, x_train, y_train, x_val, y_val, model_save_path):
    start = time()
    input_img = tf.keras.Input(shape=input_shape)
   
    # First Convolutional Block
    Z11 = tfl.Conv1D(filters=16, kernel_size=7, strides=1, padding='same')(input_img)
    Z12 = tfl.Conv1D(filters=8, kernel_size=3, strides=1, padding='same')(Z11)
    A1 = tfl.ReLU()(Z12)
    P1 = tfl.MaxPooling1D(pool_size=12, padding='same')(A1)
    D2 = tfl.Dropout(0.2)(P1)
    # Second Convolutional Block
    Z2 = tfl.Conv1D(filters=12, kernel_size=4, strides=1, padding='same')(D2)
    A2 = tfl.ReLU()(Z2)
    P2 = tfl.MaxPooling1D(pool_size=6, padding='same')(A2)

    # Flatten and Dense layers
    F = tfl.Flatten()(P2)
    op = tfl.Dense(units=16, activation='relu')(F)
    outputs = tfl.Dense(units=1, activation='sigmoid')(op)

    model = tf.keras.Model(inputs=input_img, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', Recall(name='recall')])
    model.summary()

    # Model checkpoint to save best model based on validation accuracy
    checkpoint = ModelCheckpoint(
        model_save_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=30, verbose=2, restore_best_weights=True)
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=150,
        batch_size=75,
        shuffle=True,
        callbacks=[early_stopping, checkpoint]
    )
    
    t = time() - start
    print('Total training time:', t)
    history.history['training_time'] = t
    
    return model, history

# Load training file
# change this for each health condition in the balanced dataset (healthy/ins/narco...) :
train_file = r"data\balanced\bal_healthy.csv" 

# Load the training data
data = pd.read_csv(train_file).values

# Split into input & output
x_combined = data[:, :-1]  # All columns except the last one
y_combined = data[:, -1]   # The last column

# Transform the target column to 0 & 1 classification
y_combined = np.where(y_combined == 0, 0, 1)

# Split into train, validation, and test sets
x_train, x_temp, y_train, y_temp = train_test_split(x_combined, y_combined, test_size=0.3, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.666, random_state=42)

# define file paths
# change both the folder name and the file name according to the health condition of the train_file
train_set_path = r"data\splitted-data\healthy\healthy_train_set22.csv"
validation_set_path = r"data\splitted-data\healthy\healthy_validation_set22.csv"
test_set_path = r"data\splitted-data\healthy\healthy_test_set22.csv"

# Ensure the directory exists before saving the dataset splits
os.makedirs(os.path.dirname(train_set_path), exist_ok=True)

# Save datasets to CSV
pd.DataFrame(np.column_stack((x_train, y_train))).to_csv(train_set_path, index=False, header=True)
pd.DataFrame(np.column_stack((x_val, y_val))).to_csv(validation_set_path, index=False, header=True)
pd.DataFrame(np.column_stack((x_test, y_test))).to_csv(test_set_path, index=False, header=True)

# Reshape data for Conv1D
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

# Path to save the best model
model_save_path = r"models\healthy_model22.keras" # change name according to the health condition

# Train the model
conv_model, history = CAP_phase_model((x_train.shape[1], 1), x_train, y_train, x_val, y_val, model_save_path)

# Load the best saved model
best_model = tf.keras.models.load_model(model_save_path)

# Evaluate the best saved model on training, validation, and test data
train_loss, train_accuracy, train_recall = best_model.evaluate(x_train, y_train, verbose=2)
val_loss, val_accuracy, val_recall = best_model.evaluate(x_val, y_val, verbose=2)
test_loss, test_accuracy, test_recall = best_model.evaluate(x_test, y_test, verbose=2)

print(f"Best Model Training Accuracy: {train_accuracy:.2f}")
print(f"Best Model Training Recall: {train_recall:.2f}")
print(f"Best Model Training Loss: {train_loss:.2f}")

print(f"Best Model Validation Accuracy: {val_accuracy:.2f}")
print(f"Best Model Validation Recall: {val_recall:.2f}")
print(f"Best Model Validation Loss: {val_loss:.2f}")

print(f"Best Model Test Accuracy: {test_accuracy:.2f}")
print(f"Best Model Test Recall: {test_recall:.2f}")
print(f"Best Model Test Loss: {test_loss:.2f}")
