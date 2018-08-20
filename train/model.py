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
    def cnn_layer(x, filters, kernel_size, stride=1, dropout=0):
        x = Conv2D(filters, kernel_size, strides=(stride, stride), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        if dropout:
            x = Dropout(dropout)(x)
        return x

    n_boxes = shape[0] // 4

    x_in = Input(shape)

    x = x_in

    x = cnn_layer(x, 32, 5, stride=2)
    x = cnn_layer(x, 32, 3)

    x = MaxPool2D(2)(x)

    x = cnn_layer(x, 64, 3, dropout=0.2)
    x = cnn_layer(x, 128, 3, dropout=0.3)

    x = Conv2D(1, 1)(x)

    x = Flatten()(x)
    x = Softmax()(x)
    x = Reshape((n_boxes, n_boxes))(x)

    x_out = x

    model = KModel(inputs=x_in, outputs=x_out)
    opt = Adam(lr=1e-3)
    model.compile(loss=WeightedBinaryCrossentropy(500), optimizer=opt)

    return model

def median_loc_err(dataset, y_pred):
    true_locs = np.concatenate([video_data.locs for video_data in dataset.videos.values()],
                               axis=0)
    pred_locs = plane_to_loc(y_pred)

    diff = true_locs - pred_locs
    return np.median(np.sqrt(np.sum(diff**2, axis=1)))

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

        self.trained_epochs = 0
        self.learning_rates = []

    def train(self, lr, epochs, **kwargs):
        K.set_value(self.keras_model.optimizer.lr, lr)

        self.keras_model.fit_generator(iter(self.train_dataset),
                                       steps_per_epoch=len(self.train_dataset),
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

        dataset = Dataset(video_nums)

        y_true = np.concatenate(list(dataset.output_generator()), axis=0)
        y_pred = self.keras_model.predict_generator(dataset.input_generator(),
                                                    steps=len(dataset), verbose=1)

        ap_score = average_precision_score(y_true.flatten(), y_pred.flatten())
        median_loc_err_score = median_loc_err(dataset, y_pred)

        print('AP: {}'.format(ap_score))
        print('MEDIAN LOC_ERR: {}'.format(median_loc_err_score))

        return y_true, y_pred
