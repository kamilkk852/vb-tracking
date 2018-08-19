import cv2
import numpy as np
from keras.layers import Input, Conv2D, MaxPool2D, BatchNormalization, \
LeakyReLU, Flatten, Softmax, Reshape, Dropout
from keras.models import Model as KModel
from keras.optimizers import Adam
from train.dataset import DEFAULT_PARAMS
from train.data_processing import plane_to_loc, get_frames, get_input
from train.model import WeightedBinaryCrossentropy
from copy import copy
import particle_filter

class Video():
    def __init__(self, video_path, frame_nums, freq):
        self.video_path = video_path
        self.freq = freq
        self.frame_nums = frame_nums
        self.params = copy(DEFAULT_PARAMS)

        cap = cv2.VideoCapture(self.video_path)
        self.video_shape = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))

    def process(self, **kwargs):
        self.params['chunk_size'] = kwargs.get('chunk_size', DEFAULT_PARAMS['chunk_size'])
        self.params['speed_cutoff'] = kwargs.get('speed_cutoff', DEFAULT_PARAMS['speed_cutoff'])
        self.params['input_size'] = kwargs.get('input_size', DEFAULT_PARAMS['input_size'])
        self.params['n_boxes'] = kwargs.get('n_boxes', DEFAULT_PARAMS['n_boxes'])
        self.params['n_diffs'] = kwargs.get('n_diffs', DEFAULT_PARAMS['n_diffs'])
        self.params['step'] = int(np.ceil(0.25 / self.params['speed_cutoff'] * self.freq))

        frames = get_frames(self.video_path, self.frame_nums, self.params['input_size'])
        X = get_input(frames, self.params['step'], self.params['n_diffs'])

        return X

def get_model(weights_path, **kwargs):
    input_size = kwargs.get('input_size', 200)
    n_diffs = kwargs.get('n_diffs', 1)
    shape = (input_size, input_size, 6 + 6*n_diffs)

    x_in = Input(shape)

    x = x_in

    x = Conv2D(32, 5, strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(32, 3, strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = MaxPool2D(2)(x)

    x = Conv2D(64, 3, strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(0.2)(x)

    x = Conv2D(128, 3, strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(0.3)(x)

    x = Conv2D(1, 1)(x)

    x = Flatten()(x)
    x = Softmax()(x)
    x = Reshape((input_size // 4, input_size // 4))(x)

    x_out = x

    model = KModel(inputs=x_in, outputs=x_out)
    opt = Adam(lr=1e-3)
    model.compile(loss=WeightedBinaryCrossentropy(500), optimizer=opt)

    model.load_weights(weights_path)

    return model

def no_move_model(cnn_preds):
    n = cnn_preds.shape[1]
    argw = np.argwhere(cnn_preds)

    argwhere_to_loc = lambda argw: (np.array(argw) + 0.5) / n

    preds = []

    for i in range(argw.shape[0] + 1):
        loc = argwhere_to_loc(argw[i - 1, 1:])
        if i == 0:
            loc = argwhere_to_loc(argw[i, 1:])
            preds += [loc]*argw[i, 0]
        elif i < argw.shape[0]:
            preds += [loc]*(argw[i, 0] - argw[i - 1, 0])
        else:
            preds += [loc]*(cnn_preds.shape[0] - argw[i - 1, 0])

    preds = np.array(preds)

    return preds

def linear_model(cnn_preds):
    n = cnn_preds.shape[1]
    argw = np.argwhere(cnn_preds)

    argwhere_to_loc = lambda argw: (np.array(argw) + 0.5) / n

    preds = []

    for i in range(argw.shape[0] + 1):
        if i == 0:
            loc = argwhere_to_loc(argw[i, 1:])
            preds += [loc]*argw[i, 0]
        elif i < argw.shape[0]:
            loc1 = argwhere_to_loc(argw[i - 1, 1:])
            loc2 = argwhere_to_loc(argw[i, 1:])
            diff = loc2 - loc1
            step = diff / (argw[i, 0] - argw[i - 1, 0])
            preds += [loc1 + j*step for j in range(argw[i, 0] - argw[i - 1, 0])]
        else:
            loc = argwhere_to_loc(argw[i - 1, 1:])
            preds += [loc]*(cnn_preds.shape[0] - argw[i - 1, 0])

    preds = np.array(preds)

    return preds

def particle_model(X, cnn_preds):
    argw = np.argwhere(cnn_preds)
    input_size = X.shape[1]
    n = cnn_preds.shape[1]

    if argw.shape[0] == 0:
        return np.zeros(cnn_preds.shape[0])

    preds = []

    acc_diff = (X[:, :, :, 3]*255).astype(np.int32)

    argwhere_to_loc = lambda argw: ((np.array(argw) + 0.5) / n * input_size).astype(int)
    get_sequence = lambda st, end: list(acc_diff[st:end])

    x0 = argwhere_to_loc(argw[0, 1:])
    seq = list(reversed(get_sequence(0, argw[0, 0] + 1)))
    pred_pos = particle_filter.predict(x0, seq)
    preds += list(reversed(pred_pos[1:]))

    for i in range(argw.shape[0]):
        st = argw[i, 0]
        if i < argw.shape[0] - 1:
            end = argw[i + 1, 0]
        else:
            end = cnn_preds.shape[0]

        seq = get_sequence(st, end)
        x0 = argwhere_to_loc(argw[i, 1:])
        pred_pos = particle_filter.predict(x0, seq)
        preds += pred_pos

    preds = np.array(preds) / input_size
    return preds

def predict(video_path, frame_nums, freq, weights_path='train/weights.h5', **kwargs):
    model = kwargs.get('model', 'particle_model')
    threshold = kwargs.get('threshold', 0.75)

    video = Video(video_path, frame_nums, freq)
    X = video.process(**kwargs)

    cnn_model = get_model(weights_path)
    cnn_preds = cnn_model.predict(X, batch_size=32, verbose=0)

    if model == 'cnn_model':
        return X[:, :, :, :3], plane_to_loc(cnn_preds)
    elif model == 'no_move_model':
        return X[:, :, :, :3], no_move_model(cnn_preds > threshold)
    elif model == 'linear_model':
        return X[:, :, :, :3], linear_model(cnn_preds > threshold)
    elif model == 'particle_model':
        return X[:, :, :, :3], particle_model(X, cnn_preds > threshold)

    return None

def short_predict(cnn_preds, X=None, **kwargs):
    model = kwargs.get('model', 'particle_model')
    threshold = kwargs.get('threshold', 0.75)

    if model == 'cnn_model':
        return plane_to_loc(cnn_preds)
    elif model == 'no_move_model':
        return no_move_model(cnn_preds > threshold)
    elif model == 'linear_model':
        return linear_model(cnn_preds > threshold)
    elif model == 'particle_model':
        return particle_model(X, cnn_preds > threshold)

    return None
