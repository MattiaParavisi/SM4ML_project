import os
from PIL import Image
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold, StratifiedKFold
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import SGD
from keras import backend as K
from tensorflow.image import rgb_to_grayscale, grayscale_to_rgb
import numpy as np
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

image_height = 225
image_width = 240
batch_size = 32
epochs_numb = 10

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

validation_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        './../../../NN_data/train',
        target_size=(image_height, image_width),
        batch_size=batch_size,
        class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
        './../../../NN_data/val',
        target_size=(image_height, image_width),
        batch_size=batch_size,
        class_mode='binary')

test_generator = test_datagen.flow_from_directory(
    './../../../NN_data/test',
    target_size=(image_height,image_width),
    batch_size=batch_size,
    class_mode='binary')

epochs_numb = 10

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(image_height, image_width, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.50))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

hist1 = model.fit(
        train_generator,
        steps_per_epoch=trainig_size // batch_size,
        epochs=epochs_numb,
        validation_data=validation_generator,
        validation_steps=validation_size // batch_size)

epochs_numb = 15

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(image_height, image_width, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.50))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

hist2 = model.fit(
        train_generator,
        steps_per_epoch=trainig_size // batch_size,
        epochs=epochs_numb,
        validation_data=validation_generator,
        validation_steps=validation_size // batch_size)

epochs_numb = 30

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(image_height, image_width, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.50))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

hist3 = model.fit(
        train_generator,
        steps_per_epoch=trainig_size // batch_size,
        epochs=epochs_numb,
        validation_data=validation_generator,
        validation_steps=validation_size // batch_size)

#adapt list length to the longest

temp_hist1 = hist1

for key in list(hist1.history.keys()):
    for i in range(len(hist3.history[key]) - len(hist1.history[key])):
        temp_hist1.history[key].append(hist1.history[key][-1])

temp_hist2 = hist2

for key in list(hist2.history.keys()):
    for i in range(len(hist3.history[key]) - len(hist2.history[key])):
        temp_hist2.history[key].append(hist2.history[key][-1])

#plot results

plt.plot(temp_hist1.history['accuracy'])
plt.plot(temp_hist2.history['accuracy'])
plt.plot(hist3.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['model 1 accuracy', 'model 2 accuracy', 'model 3 accuracy'], loc='lower right')
plt.show()
plt.savefig('epochs_numb_analysis_accuracy.png')
plt.clf()

plt.plot(temp_hist1.history['val_accuracy'])
plt.plot(temp_hist2.history['val_accuracy'])
plt.plot(hist3.history['val_accuracy'])
plt.title('model validation accuracy')
plt.ylabel('validation accuracy')
plt.xlabel('epoch')
plt.legend(['model 1 validation accuracy', 'model 2 validation accuracy', 'model 3 validation accuracy'], loc='upper right')
plt.show()
plt.savefig('epochs_numb_analysis_val_accuracy.png')
plt.clf()

plt.plot(temp_hist1.history['loss'])
plt.plot(temp_hist2.history['loss'])
plt.plot(hist3.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['model 1 loss', 'model 2 loss', 'model 3 loss'], loc='lower right')
plt.show()
plt.savefig('epochs_numb_analysis_loss.png')
plt.clf()

plt.plot(temp_hist1.history['val_loss'])
plt.plot(temp_hist2.history['val_loss'])
plt.plot(hist3.history['val_loss'])
plt.title('model validation loss')
plt.ylabel('validation loss')
plt.xlabel('epoch')
plt.legend(['model 1 validation loss', 'model 2 validation loss', 'model 3 validation loss'], loc='upper right')
plt.show()
plt.savefig('epochs_numb_analysis_val_loss.png')
