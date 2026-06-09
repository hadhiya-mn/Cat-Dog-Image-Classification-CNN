import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# =========================
# Data Preparation
# =========================

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    'dataset',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    'dataset',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Display class mapping
print("Class Indices:", train_generator.class_indices)

# =========================
# CNN Model
# =========================

model = tf.keras.Sequential([

    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation='relu',
        input_shape=(128, 128, 3)
    ),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation='relu'
    ),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(
        128,
        (3, 3),
        activation='relu'
    ),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(
        128,
        activation='relu'
    ),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(
        1,
        activation='sigmoid'
    )
])

# =========================
# Compile Model
# =========================

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# =========================
# Train Model
# =========================

history = model.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator
)

# =========================
# Accuracy Graph
# =========================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history['accuracy'],
    label='Training Accuracy'
)

plt.plot(
    history.history['val_accuracy'],
    label='Validation Accuracy'
)

plt.title('Training vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.savefig('accuracy_graph.png')
plt.show()

# =========================
# Loss Graph
# =========================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history['loss'],
    label='Training Loss'
)

plt.plot(
    history.history['val_loss'],
    label='Validation Loss'
)

plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.savefig('loss_graph.png')
plt.show()

# =========================
# Evaluate Model
# =========================

loss, accuracy = model.evaluate(val_generator)

print(f"\nValidation Accuracy: {accuracy * 100:.2f}%")

# =========================
# Save Model
# =========================

model.save('cat_dog_model.h5')

print("\n✅ Model training complete!")
print("✅ Model saved as: cat_dog_model.h5")
print("✅ Accuracy graph saved as: accuracy_graph.png")
print("✅ Loss graph saved as: loss_graph.png")