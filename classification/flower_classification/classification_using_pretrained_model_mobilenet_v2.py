# imports
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import numpy as np
import matplotlib.pyplot as plt

# tensorflow imports 
import tensorflow as tf
tf.enable_eager_execution()
tf.logging.set_verbosity(tf.logging.ERROR)

import tensorflow_datasets as tfds
import tensorflow_hub as hub
from tensorflow.keras import layers

print("tensorflow version: {}".format(tf.__version__))

splits = tfds.Split.TRAIN.subsplit([70, 30])
(training_set, validation_set), dataset_info = tfds.load('tf_flowers', with_info=True, as_supervised=True, split=splits)

num_classes = dataset_info.features['label'].num_classes
num_examples = dataset_info.splits['train'].num_examples

num_train_examples = 0
num_val_examples = 0

for example in training_set:
  num_train_examples += 1
  
for example in validation_set:
  num_val_examples += 1

print('No. of classes: {}'.format(num_classes))
print('No. of examples: {}'.format(num_examples))
print('No. of validation examples: {}'.format(num_train_examples))
print('No. of validation examples: {}'.format(num_val_examples))

for i, examples in enumerate(training_set.take(5)):
  print('Image {} shape: {}, label: {}'.format(i + 1, examples[0].shape, examples[1]))


# format images
IMAGE_RES = 224

def format_image(image, label):
  image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES))/255.0
  return image, label

BATCH_SIZE = 32

train_batches = training_set.shuffle(num_train_examples//4).map(format_image).batch(BATCH_SIZE).prefetch(1) 
val_batches = validation_set.map(format_image).batch(BATCH_SIZE).prefetch(1)

# train model
MOBILE_NET_V2_URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2"
feature_extractor = hub.KerasLayer(MOBILE_NET_V2_URL, input_shape=(IMAGE_RES, IMAGE_RES, 3))
feature_extractor.trainable = False

model = tf.keras.Sequential([
    feature_extractor,
    layers.Dense(num_classes, activation='softmax')
])
model.summary()

epochs = 6
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(
    train_batches,
    epochs=epochs, 
    validation_data=val_batches
)

# Plot Training and Validation Graphs.
acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training vs Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend('upper right')
plt.title('Training vs validation loss')

plt.show()


# Check Predictions
class_names = np.array(dataset_info.features['label'].names)
print(class_names)

# Create an Image Batch and Make Predictions
image_batch, label_batch = next(iter(train_batches))

image_batch = image_batch.numpy()
label_batch = label_batch.numpy()

predicted_batch = model.predict(image_batch)
predicted_batch = tf.squeeze(predicted_batch).numpy()

predicted_ids = np.argmax(predicted_batch, axis=-1)
predicted_classes = class_names[predicted_ids]

print(predicted_classes)

plt.figure(figsize=(10, 9))

for n in range(30):
  plt.subplot(6, 5, n + 1)
  plt.imshow(image_batch[n])
  color = 'green' if predicted_ids[n] == label_batch[n] else 'red'
  plt.title(predicted_classes[n].title(), color= color)
  plt.axis('off')
_ = plt.suptitle('Model predictions (green: correct, red: incorrect)')
