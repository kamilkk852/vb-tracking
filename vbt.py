import cv2
import numpy as np
from train.dataset import DEFAULT_PARAMS
from train.data_processing import plane_to_loc, get_frames, get_input
from train.model import get_architecture
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
        for arg in kwargs:
            self.params[arg] = kwargs.get(arg)

        self.params['step'] = int(np.ceil(0.25 / self.params['speed_cutoff'] * self.freq))

        frames = get_frames(self.video_path, self.frame_nums, self.params['input_size'])
        X = get_input(frames, self.params['step'], self.params['n_diffs'])

        return X

def get_model(**kwargs):
    weights_path = kwargs.get('weights_path', 'train/weights.h5')
    input_size = kwargs.get('input_size', 200)
    n_diffs = kwargs.get('n_diffs', 1)
    shape = (input_size, input_size, 6 + 6*n_diffs)

    model = get_architecture(shape)
    model.load_weights(weights_path)

    return model

def no_move_model(cnn_preds):
    n = cnn_preds.shape[1]
    argw = np.argwhere(cnn_preds)
    argwhere_to_loc = lambda argw: (np.array(argw) + 0.5) / n

    preds = np.zeros((cnn_preds.shape[0], 2))

    preds[:argw[0, 0]] = argwhere_to_loc(argw[0, 1:])

    for i in range(1, argw.shape[0]):
        preds[argw[i-1, 0]:argw[i, 0]] = argwhere_to_loc(argw[i-1, 1:])

    preds[argw[-1, 0]:] = argwhere_to_loc(argw[-1, 1:])

    return preds

def linear_model(cnn_preds):
    n = cnn_preds.shape[1]
    argw = np.argwhere(cnn_preds)
    argwhere_to_loc = lambda argw: (np.array(argw) + 0.5) / n

    preds = np.zeros((cnn_preds.shape[0], 2))

    preds[:argw[0, 0]] = argwhere_to_loc(argw[0, 1:])

    for i in range(1, argw.shape[0]):
        loc1, loc2 = map(argwhere_to_loc, argw[i-1:i, 1:])
        n_frames = argw[i, 0] - argw[i - 1, 0]
        step = (loc2 - loc1) / n_frames
        preds[argw[i-1, 0]:argw[i, 0]] = [loc1 + j*step for j in range(n_frames)]

    preds[argw[-1, 0]:] = argwhere_to_loc(argw[-1, 1:])

    return preds

def particle_model(acc_diff, cnn_preds):
    argw = np.argwhere(cnn_preds)
    input_size = acc_diff.shape[1]
    n = cnn_preds.shape[1]

    preds = np.zeros((cnn_preds.shape[0], 2))

    if argw.shape[0] == 0:
        return preds

    argwhere_to_pixel_loc = lambda argw: ((np.array(argw) + 0.5) / n * input_size).astype(int)
    get_sequence = lambda st, end: list(acc_diff[st:end])

    start_frame = argw[0, 0]
    x0 = argwhere_to_pixel_loc(argw[0, 1:])
    seq = get_sequence(0, start_frame + 1)
    preds[:start_frame] = particle_filter.predict(x0, seq, reverse=True)[:-1]

    for i in range(argw.shape[0]):
        start_frame = argw[i, 0]
        if i < argw.shape[0] - 1:
            end_frame = argw[i + 1, 0]
        else:
            end_frame = cnn_preds.shape[0]

        x0 = argwhere_to_pixel_loc(argw[i, 1:])
        seq = get_sequence(start_frame, end_frame)
        preds[start_frame:end_frame] = particle_filter.predict(x0, seq)

    preds /= input_size
    return preds

def predict(video_path, frame_nums, **kwargs):
    model = kwargs.get('model', 'particle_model')
    threshold = kwargs.get('threshold', 0.75)
    freq = kwargs.get('freq', 120)

    video = Video(video_path, frame_nums, freq)
    X = video.process(**kwargs)

    cnn_model = get_model(**kwargs)
    cnn_preds = cnn_model.predict(X, batch_size=32, verbose=0)

    frames = X[:, :, :, :3]

    if model == 'cnn_model':
        return frames, plane_to_loc(cnn_preds)
    elif model == 'no_move_model':
        return frames, no_move_model(cnn_preds > threshold)
    elif model == 'linear_model':
        return frames, linear_model(cnn_preds > threshold)
    elif model == 'particle_model':
        acc_diff = (X[:, :, :, 3]*255).astype(np.int32)
        return frames, particle_model(acc_diff, cnn_preds > threshold)

    return None

def short_predict(cnn_preds, **kwargs):
    model = kwargs.get('model', 'particle_model')
    threshold = kwargs.get('threshold', 0.75)

    if model == 'cnn_model':
        return plane_to_loc(cnn_preds)
    elif model == 'no_move_model':
        return no_move_model(cnn_preds > threshold)
    elif model == 'linear_model':
        return linear_model(cnn_preds > threshold)
    elif model == 'particle_model':
        acc_diff = (kwargs.get('acc_diff')*255).astype(np.int32)
        return particle_model(acc_diff, cnn_preds > threshold)

    return None
