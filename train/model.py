import numpy as np
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPool2D, BatchNormalization, \
LeakyReLU, Flatten, Softmax, Reshape, Dropout
from keras.models import Model as KModel
from keras.optimizers import Adam
from keras import backend as K
from sklearn.metrics import average_precision_score
from train.dataset import Dataset
from train.data_processing import plane_to_loc

def get_architecture(shape):
    def cnn_layer(x, filters, kernel_size, stride, dropout=0):
        x = Conv2D(filters, kernel_size, strides=(stride, stride), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        if dropout:
            x = Dropout(dropout)(x)
        return x

    n_boxes = shape[0] // 4

    x_in = Input(shape)

    x = x_in

    x = cnn_layer(x, 32, 5, 2)
    x = cnn_layer(x, 32, 3, 1)

    x = MaxPool2D(2)(x)

    x = cnn_layer(x, 64, 3, 1, dropout=0.2)
    x = cnn_layer(x, 128, 3, 1, dropout=0.3)

    x = Conv2D(1, 1)(x)

    x = Flatten()(x)
    x = Softmax()(x)
    x = Reshape((n_boxes, n_boxes))(x)

    x_out = x

    model = KModel(inputs=x_in, outputs=x_out)
    opt = Adam(lr=1e-3)
    model.compile(loss=WeightedBinaryCrossentropy(500), optimizer=opt)

    return model

class WeightedBinaryCrossentropy(float):
    def __call__(self, y_true, y_pred):
        return tf.keras.backend.binary_crossentropy(y_true, y_pred) + \
                (self-1)*tf.keras.backend.binary_crossentropy(y_true, y_true*y_pred)

class CNNModel():
    def __init__(self, train_vids, valid_vids, print_sum=True):
        self.train_vids = train_vids
        self.valid_vids = valid_vids

        self.train_dataset = Dataset(train_vids)
        self.all_dataset = Dataset(train_vids + valid_vids)

        input_size = self.train_dataset.params['input_size']
        n_diffs = self.train_dataset.params['n_diffs']
        shape = (input_size, input_size, 6 + 6*n_diffs)
        self.keras_model = get_architecture(shape)
        if print_sum:
            print(self.keras_model.summary())

        self.train_steps = sum(self.train_dataset.videos[video_num].chunk_cnt for video_num in train_vids)

        self.trained_epochs = 0
        self.learning_rates = []

    def train(self, lr, epochs, **kwargs):
        K.set_value(self.keras_model.optimizer.lr, lr)

        self.keras_model.fit_generator(iter(self.train_dataset),
                                       steps_per_epoch=self.train_steps,
                                       epochs=epochs, **kwargs)

        self.trained_epochs += epochs
        self.learning_rates += [lr]*epochs

    def save_weights(self, filename):
        self.keras_model.save_weights(filename)

    def evaluate(self, video_nums='valid'):
        if video_nums == 'valid':
            video_nums = self.valid_vids
        elif video_nums == 'train':
            video_nums = self.train_vids

        steps = sum(self.all_dataset.videos[video_num].chunk_cnt for video_num in video_nums)

        def generator(i):
            for video_num in video_nums:
                for chunk_num in range(self.all_dataset.videos[video_num].chunk_cnt):
                    yield self.all_dataset.videos[video_num][chunk_num][i]

        y_true = np.concatenate(list(generator(1)), axis=0)
        y_pred = self.keras_model.predict_generator(generator(0), steps=steps, verbose=1)

        ap_score = average_precision_score(y_true.flatten(), y_pred.flatten())

        true_locs = np.concatenate([self.all_dataset.videos[video_num].locs \
                                    for video_num in video_nums], axis=0)
        pred_locs = plane_to_loc(y_pred)

        diff = true_locs - pred_locs

        loc_err_score = np.median(np.sqrt(diff[:, 0]**2 + diff[:, 1]**2))

        print('AP: {}'.format(ap_score))
        print('MEDIAN LOC_ERR: {}'.format(loc_err_score))

        return y_true, y_pred
