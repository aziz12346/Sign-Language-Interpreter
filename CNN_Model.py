import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt



# Load dataset
input_file = r"C:\Development\leapc-python-bindings-main\Dataset\Handsigns_preprocessed.csv"
df = pd.read_csv(input_file, header=None)

print("Dataset shape:", df.shape)


# Separate labels and features
y = df.iloc[:, 0].values.astype("int32")
X = df.iloc[:, 4:].values.astype("float32")

print("Features shape:", X.shape)
print("Label shape:", y.shape)


# Train/test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# Prepare features (X) and labels (y) for training a 1D CNN
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)


# 1D CNN Model -> CHECK FILTER VALUES!
num_classes = len(np.unique(y))
input_shape = (X_train.shape[1], 1)

model = keras.Sequential([
    keras.Input(shape=input_shape),

    # First Layer 
    layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
    layers.MaxPooling1D(pool_size=2),

    # Second Layer
    layers.Conv1D(filters=128, kernel_size=3, activation='relu'),
    layers.MaxPooling1D(pool_size=2),

    # Flatten Layer
    layers.Flatten(),

    # Fully Connected Dense Layer
    layers.Dense(64, activation='relu'),

    # Dropout Layer
    layers.Dropout(0.3),

    # Fully Connected (Output) Layer
    layers.Dense(num_classes, activation='softmax')
])


# Compile the Model -> CHECK LOSS!
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
)

model.summary()


# Callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)


# Train the CNN Model -> CHECK VALIDATION!
history = model.fit(X_train, y_train, epochs=30, batch_size=16, verbose=1, validation_split=0.2, callbacks=[early_stopping])



# Evaluate the Model Performance
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Confusion Matrix
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()


# Training Performance Visualised
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('CNN Training Accuracy Over Epochs')
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('CNN Training Loss Over Epochs')
plt.show()


# Save the Model
model.save(r"C:\Development\leapc-python-bindings-main\Dataset\BSL_CNN.keras")
print("Model saved successfully.")
