import os
import json

from comet_ml import Experiment

import numpy as np
import tensorflow as tf
cifar10 = tf.keras.datasets.cifar10
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator

global params

def build_graph(x, is_training):
    weight_decay = 1e-4
    regularizer = tf.contrib.layers.l2_regularizer(scale=weight_decay)

    x = tf.layers.conv2d(x,
                         32,
                         (3,3),
                         padding='SAME',
                         kernel_regularizer=regularizer)
    x = tf.nn.elu(x)
    x = tf.layers.batch_normalization(x, training=is_training)

    x = tf.layers.conv2d(x,
                         32,
                         (3,3),
                         padding='SAME',
                         kernel_regularizer=regularizer)
    x = tf.nn.elu(x)
    x = tf.layers.batch_normalization(x, training=is_training)
    x = tf.layers.max_pooling2d(x, (2,2), (1,1))
    x = tf.layers.dropout(x, rate=0.2, training=is_training)

    x = tf.layers.conv2d(x,
                         64,
                         (3,3),
                         padding='SAME',
                         kernel_regularizer=regularizer)
    x = tf.nn.elu(x)
    x = tf.layers.batch_normalization(x, training=is_training)
    x = tf.layers.conv2d(x,
                         32,
                         (3,3),
                         padding='SAME',
                         kernel_regularizer=regularizer)
    x = tf.nn.elu(x)
    x = tf.layer.batch_normalization(x, training=is_training)
    x = tf.layers.max_pooling2d(x, (2,2), (1,1))
    x = tf.layers.dropout(x, rate=0.3, training=is_training)

    x = tf.layers.conv2d(x,
                         128,
                         (3,3),
                         padding='SAME',
                         kernel_regularizer=regularizer)
    x = tf.nn.elu(x)
    x = tf.layers.batch_normalization(x, training=is_training)
    x = tf.layers.conv2d(x,
                         128,
                         (3,3),
                         padding='SAME',
                         kernel_regularizer=regularizer)
    x = tf.nn.elu(x)
    x = tf.layers.batch_normalization(x, training=is_training)
    x = tf.layers.max_pooling2d(x, (2,2), (1,1))
    x = tf.layers.dropout(x, rate=0.4, training=is_training)

    x = tf.layers.flatten(x)
    x = tf.layers.dense(x, 10)

    return x


def get_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    #z-score
    mean = np.mean(x_train,axis=(0,1,2,3))
    std = np.std(x_train,axis=(0,1,2,3))
    x_train = (x_train-mean)/(std+1e-7)
    x_test = (x_test-mean)/(std+1e-7)

    return (x_train, y_train), (x_test, y_test)

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

def data_schedule_to_str(ds):
    ds_string = ''
    for f, e in ds:
        ds_string += '{} for {} epochs\n'.format(round(f,5), e)

    return ds_string

def fraction(a, ratio):
    r = int(ratio*a.shape[0])
    return a.copy()[:r]

#https://github.com/tensorflow/tensorflow/issues/4814#issuecomment-314801758
def create_reset_metric(metric, scope='reset_metrics', **metric_args):
    with tf.variable_scope(scope) as scope:
        metric_op, update_op = metric(**metric_args)
        vs = tf.get_collection(collection=tf.GraphKeys.LOCAL_VARIABLES,
                               scope=scope)
        reset_op = tf.variables_initializer(vs)
    return metric_op, update_op, reset_op

def get_learning_rate(epoch):
    lrate = 0.001
    if epoch > 75:
        lrate = 0.0005
    elif epoch > 100:
        lrate = 0.0003
    return lrate

def main():
    batch_size = 2048
    #data_schedule = [(1/32, 2), (1/16, 4), (1/8, 8), (1/4, 16), (1/2, 32), (1, 64)]
    data_schedule = [(1, 126)]

    log_params = {
        'batch_size': batch_size,
        'data_schedule': data_schedule_to_str(data_schedule)
    }

    experiment = load_comet()
    experiment.log_multiple_params(log_params)

    learning_rate = tf.placeholder(tf.float32)
    x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
    y = tf.placeholder(tf.float32, shape=[None, 1])
    is_training = tf.placeholder(tf.bool)

    net = build_graph(x, is_training)


    loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=net,
                                                      labels=tf.one_hot(y, 10))
    train = tf.train.RMSPropOptimizer(learning_rate, decay=1e-6).minimize(loss)
    acc, acc_update, acc_reset = create_reset_metric(tf.metrics.accuracy,
                                                     'metric_acc',
                                                     labels=y,
                                                     predictions=tf.argmax(net))

    (x_train, y_train), (x_test, y_test) = get_data()
    train_gen = ImageDataGenerator(rotation_range=15,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   horizontal_flip=True)

    test_gen = ImageDataGenerator()

    num_train_batches = 50e3 // batch_size
    num_test_batches = 10e3 // batch_size

    init = tf.global_variables_initializer()


    with tf.Session() as sess:
        sess.run(init)
        experiment.set_model_graph(sess.graph)

        current_epoch = 0
        for ratio, num_epochs in data_schedule:
            train_data = train_gen.flow(fraction(x_train, ratio),
                                        fraction(y_train, ratio),
                                        batch_size=batch_size,
                                        shuffle=True)

            test_data = test_gen.flow(x_test, y_test, batch_size=batch_size)

            for _ in range(num_epochs):
                current_epoch += 1

                sess.run(acc_reset)
                # training
                with experiment.train():
                    experiment.log_metric('data_ratio',
                                          ratio,
                                          step=current_epoch)
                    experiment.log_metric('learning_rate',
                                          get_learning_rate(current_epoch),
                                          step=current_epoch)
                    for _ in range(num_train_batches):
                        batch_xs, batch_ys = next(train_data)
                        feed_dict = {x:batch_xs,
                                     y:batch_ys,
                                     is_training:True,
                                     learning_rate:get_learning_rate(current_epoch)}
                        sess.run([train, acc_update], feed_dict=feed_dict)

                    experiment.log_metric('accuracy',
                                          sess.run(acc),
                                          step=current_epoch)

                sess.run(acc_reset)
                with experiment.test():
                    for _ in range(num_test_batches):
                        batch_xs, batch_ys = next(test_data)
                        feed_dict = {x:batch_xs, y:batch_ys, is_training:False}
                        sess.run([train, acc_update], feed_dict=feed_dict)

                    experiment.log_metric('accuracy',
                                          sess.run(acc),
                                          step=current_epoch)


if __name__=='__main__':
    main()