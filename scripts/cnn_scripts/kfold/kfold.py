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
n = 5
num_epochs = 10

def get_model():
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
    return model

def get_model_name(k):
    return 'model_k_fold_'+str(k)+'.h5'

def my_loss_fn(y_true, y_pred):
    casted = tf.subtract(y_true, y_pred)
    squared = tf.square(casted)
    return squared

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
delta_improv = EarlyStopping(monitor='loss',  min_delta=0.005, patience=5)

base_dir = './CatsDogs_resized'
images = []
true_labels = []
for dirs in os.listdir(base_dir):
    for img in os.listdir('/'.join([base_dir, dirs])):
        images.append('/'.join([base_dir,dirs,img]))
        true_labels.append(dirs.split('_')[0])
df = pd.DataFrame({'Name' : images, 'True Labels': true_labels})

Y = df[['True Labels']]

kf = KFold(n_splits = n)

idg = ImageDataGenerator(width_shift_range=0.1,
                         height_shift_range=0.1,
                         zoom_range=0.3,
                         fill_mode='nearest',
                         horizontal_flip = True,
                         rescale=1./255)

test_acc = []
test_loss = []

df = df.sample(frac = 1)

save_dir = './kfold_models_best/'
fold_var = 1

for train_index, val_index in kf.split(np.zeros(len(Y)),Y):
    training_data = df.iloc[train_index]
    validation_data = df.iloc[val_index]

    train_data_generator = idg.flow_from_dataframe(training_data,
                                                   x_col = "Name", y_col = "True Labels",
                                                   target_size=(image_height,image_width),
                                                   class_mode = "binary", shuffle = True)
    valid_data_generator  = idg.flow_from_dataframe(validation_data,
                                                    x_col = "Name", y_col = "True Labels",
                                                    target_size=(image_height,image_width),
                                                    class_mode = "binary", shuffle = True)
    model = get_model()
    model.compile(loss=my_loss_fn,optimizer='rmsprop',metrics=['accuracy'])
    checkpoint = ModelCheckpoint(save_dir+get_model_name(fold_var),
                                                    monitor='val_accuracy', verbose=1,
                                                    save_best_only=True, mode='max')
    callbacks_list = [checkpoint,reduce_lr,delta_improv]
    history = model.fit(train_data_generator,
                        epochs=num_epochs,
                        callbacks=callbacks_list,
                        validation_data=valid_data_generator)
    results = model.evaluate(valid_data_generator)
    results = dict(zip(model.metrics_names,results))
    test_acc.append(results['accuracy'])
    test_loss.append(results['loss'])
    tf.keras.backend.clear_session()
    fold_var += 1

print(np.mean(test_acc))
print(np.mean(test_loss))
