import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np

# Path to the FER2013 CSV file
dataset_path = 'fer2013.csv'

# Hyperparameters
batch_size = 32
img_height = 48
img_width = 48
num_classes = 7
epochs = 20

# 1. Load and preprocess the FER2013 data using pandas
df = pd.read_csv(dataset_path)
# Split by Usage column in CSV: Training, PublicTest, PrivateTest
df_train = df[df['Usage'] == 'Training']
df_val   = df[df['Usage'] == 'PublicTest']

# Helper to convert pixels string to array
def df_to_dataset(df_subset):
    # Convert pixel strings to numpy arrays
    pixels = np.vstack(df_subset['pixels'].apply(lambda s: np.fromstring(s, sep=' ')))
    # Normalize and reshape
    pixels = pixels.astype('float32') / 255.0
    pixels = pixels.reshape(-1, img_height, img_width, 1)
    # One-hot encode labels
    labels = keras.utils.to_categorical(df_subset['emotion'], num_classes)
    return pixels, labels

X_train, y_train = df_to_dataset(df_train)
X_val,   y_val   = df_to_dataset(df_val)

# 2. Build tf.data datasets
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))\
    .shuffle(buffer_size=len(X_train))\
    .batch(batch_size)\
    .prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))\
    .batch(batch_size)\
    .prefetch(tf.data.AUTOTUNE)

# 3. Define the CNN model
model = keras.Sequential([
    keras.layers.Input(shape=(img_height, img_width, 1)),
    keras.layers.Conv2D(32, 3, activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64, 3, activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(128, 3, activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(num_classes, activation='softmax')
])

# 4. Compile the model 
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 5. Train the model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# 6. Save the trained model in .keras format
model.save('emotion_detector_model.keras')
