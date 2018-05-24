import os
import json

from comet_ml import Experiment


# https://appliedmachinelearning.wordpress.com/2018/03/24/achieving-90-accuracy-in-object-recognition-task-on-cifar-10-dataset-with-keras-convolutional-neural-networks/
import numpy as np

import keras
from keras import backend as K
from keras import regularizers
from keras.callbacks import LearningRateScheduler
from keras.datasets import cifar10
from keras.layers import (Activation, BatchNormalization, Conv2D, Dense,
                          Dropout, Flatten, MaxPooling2D)
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils



def load_comet():
    comet_key = os.environ['COMET_API_KEY']
    return Experiment(api_key=comet_key,
                      project_name='reduced-dataset-cifar10')

def lr_schedule(epoch):
    lrate = 0.001
    if epoch > 75:
        lrate = 0.0005
    elif epoch > 100:
        lrate = 0.0003
    return lrate

def fraction(a, ratio):
    r = int(ratio*a.shape[0])
    return a.copy()[:r]

def train(x_train, y_train, x_test, y_test, start_epoch, end_epoch, batch_size):
    K.clear_session()

    weight_decay = 1e-4
    model = Sequential()
    model.add(Conv2D(32, (3,3),
                     padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay),
                     input_shape=x_train.shape[1:]))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3,3),
                     padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3,3),
                     padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3,3),
                     padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3,3),
                     padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3,3),
                     padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))

    model.summary()

    if 'model.h5' in os.listdir():
        model.load_weights('model.h5')

    if 'opt.json' in os.listdir():
        with open('opt.json', 'r') as f:
            opt_config = json.load(f)
        opt_rms = keras.optimizers.rmsprop.from_config(opt_config)
    else:
        opt_rms = keras.optimizers.rmsprop(lr=0.001,decay=1e-6)

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt_rms,
                  metrics=['accuracy'])


    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
    )
    datagen.fit(x_train)

    data = datagen.flow(x_train, y_train, batch_size=batch_size)

    mfit = model.fit_generator(data,
                               steps_per_epoch=50e3 // batch_size,
                               initial_epoch=start_epoch,
                               epochs=end_epoch,
                               verbose=2,
                               validation_data=(x_test,y_test),
                               callbacks=[LearningRateScheduler(lr_schedule)])
    hist = mfit.history

    with open('model.json', 'w') as json_file:
        json_file.write(model.to_json())

    model.save_weights('model.h5')

    with open('opt.json', 'w') as json_file:
        json.dump(opt_rms.get_config(), json_file)

    return hist


def data_schedule_to_str(ds):
    ds_string = ''
    for f, e in ds:
        ds_string += '{} for {} epochs\n'.format(round(f,5), e)

    return ds_string

def get_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    #z-score
    mean = np.mean(x_train,axis=(0,1,2,3))
    std = np.std(x_train,axis=(0,1,2,3))
    x_train = (x_train-mean)/(std+1e-7)
    x_test = (x_test-mean)/(std+1e-7)

    num_classes = 10
    y_train = np_utils.to_categorical(y_train,num_classes)
    y_test = np_utils.to_categorical(y_test,num_classes)

    return (x_train, y_train), (x_test, y_test)

def main():
    # params
    batch_size = 32
    data_schedule = [(1/32, 60), (1/16, 30), (1/8, 15), (1/4, 8), (1/2, 4), (1, 2)]

    log_params = {
        'batch_size': batch_size,
        'data_schedule': data_schedule_to_str(data_schedule)
    }

    # get comet setup
    experiment = load_comet()


    experiment.log_multiple_params(log_params)

    # start experiment
    (x_train, y_train), (x_test, y_test) = get_data()

    history = None
    current_epoch = 0
    for ratio, epochs in data_schedule:
        experiment.log_metric('training_ratio', ratio, step=current_epoch)
        hist = train(fraction(x_train, ratio),
                    fraction(y_train, ratio),
                    x_test,
                    y_test,
                    current_epoch,
                    current_epoch+epochs,
                    batch_size)

        if history is None:
            history = hist
        else:
            for k in hist.keys():
                history[k].extend(hist[k])

        current_epoch += epochs

    with open('model-hist.json', 'w') as f:
        json.dump(history, f)

if __name__=='__main__':
    main()
