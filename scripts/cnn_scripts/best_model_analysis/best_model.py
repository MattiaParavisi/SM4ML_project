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
epochs_numb = 300

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

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
delta_improv = EarlyStopping(monitor='loss',  min_delta=0.005, patience=5)
checkpoint_filepath = './tmp/checkpoint'
model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=True, monitor='val_accuracy', mode='max', save_best_only=True)

model=Sequential()
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(image_height, image_width, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(128,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(256,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(512,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.55))
model.add(Dense(128,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.60))
model.add(Dense(64,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.65))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',
  optimizer='rmsprop',metrics=['accuracy'])

hist = model.fit(
        train_generator,
        epochs=epochs_numb,
        validation_data=validation_generator,
        callbacks=[reduce_lr,delta_improv,model_checkpoint_callback]
        )

model.save('model_best')

model.evaluate(test_generator)

import matplotlib.pyplot as plt
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
plt.savefig('best_model_acc_val_acc.png')
plt.clf()

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
plt.savefig('best_model_loss_val_loss.png')
plt.clf()

#check for uncertain images
fig = plt.figure(figsize=(30,30))
rows = 10
columns = 10
index = 1
base_path = './../../../NN_data/test/'
for dir in os.listdir(base_path):
  i = 0
  for img in os.listdir(base_path + dir):
    img_open = Image.open(base_path + dir + '/' + img)
    img_to_array = np.array(img_open)
    img_reshaped = np.expand_dims(img_to_array, axis = 0)
    res = model.predict(img_reshaped, verbose=0)
    if res[0] < 0.60 and res[0] > 0.40:
      fig.add_subplot(rows, columns, index)
      plt.imshow(img_to_array)
      plt.title(res[0])
      index += 1
fig.show()
fig.savefig('uncertain_images.png')
