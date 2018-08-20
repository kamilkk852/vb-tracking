import cv2
import os
import os.path
import pickle
import json
import numpy as np
from glob import glob
from copy import copy
from train.data_processing import read_kva_files, get_frames, get_input, get_output, dump_chunks

SETTINGS = json.load(open('train/settings.json', 'r'))
DEFAULT_PARAMS = SETTINGS['DEFAULT_PARAMS']
DIFFS_STD = SETTINGS['DIFFS_STD']
WORKING_PATH = SETTINGS['WORKING_PATH']
FREQS = SETTINGS['FREQS']

class VideoData():
    def __init__(self, video_num):
        self.video_num = video_num
        self.video_num_str = str(video_num).zfill(4)

        self.video_path = glob('{}video{}.*'.format(WORKING_PATH, self.video_num_str))[0]
        self.kva_paths = sorted(glob('{}video{}_*.kva'.format(WORKING_PATH, self.video_num_str)))
        self.params_path = '{}data/params_{}.json'.format(WORKING_PATH, self.video_num_str)

        cap = cv2.VideoCapture(self.video_path)
        self.video_shape = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))

        self.freq = FREQS[str(self.video_num)]
        self.frame_nums, locs = read_kva_files(self.kva_paths)
        self.locs = locs / max(self.video_shape)

        self.update()

    def __getitem__(self, i):
        if isinstance(i, slice):
            if i.start is None:
                start = 0
            else: start = i.start

            if i.stop is None:
                stop = self.chunk_cnt
            else: stop = i.stop

            X, y = zip(*(pickle.load(open(self.data_paths[j], 'rb')) for j in range(start, stop)))
            X, y = map(lambda x: np.concatenate(x, axis=0), (X, y))
        else: X, y = pickle.load(open(self.data_paths[i], 'rb'))

        return X, y

    def __iter__(self):
        while True:
            chunk_num = np.random.randint(0, self.chunk_cnt)
            yield self.__getitem__(chunk_num)

    def process(self, **kwargs):
        os.makedirs('{}data'.format(WORKING_PATH), exist_ok=True)

        if self.data_paths:
            self.remove()

        for arg in kwargs:
            self.params[arg] = kwargs.get(arg)

        self.params['step'] = int(np.ceil(0.25 / self.params['speed_cutoff'] * self.freq))

        frames = get_frames(self.video_path, self.frame_nums, self.params['input_size'])
        X = get_input(frames, self.params['step'], self.params['n_diffs'])
        y = get_output(self.locs, self.params['n_boxes'])
        save_path = '{}data/data{}_'.format(WORKING_PATH, self.video_num_str)
        dump_chunks(X, y, save_path, self.params['chunk_size'])

        json.dump(self.params, open(self.params_path, 'w'))

        self.update()
        print('Data from video {} has been saved!'.format(self.video_num))

    def update(self):
        self.data_paths = sorted(glob('{}data/data{}_*'.format(WORKING_PATH,
                                                               self.video_num_str)))
        self.processed = os.path.isfile(self.params_path)
        if self.processed:
            self.params = json.load(open(self.params_path, 'r'))
        else: self.params = copy(DEFAULT_PARAMS)
        self.params['step'] = int(np.ceil(0.25 / self.params['speed_cutoff'] * self.freq))
        self.margin = self.params['step'] + self.params['n_diffs'] - 1
        self.chunk_cnt = len(self.data_paths)

    def remove(self):
        if self.processed:
            os.remove(self.params_path)

            for data_path in self.data_paths:
                os.remove(data_path)

            self.update()

class Dataset():
    def __init__(self, video_nums):
        self.videos = dict((video_num, VideoData(video_num)) for video_num in video_nums)
        assert all(video_data.processed for video_data in self.videos.values()), \
        'Some videos have not been processed yet'

        self.params = list(self.videos.values())[0].params
        del self.params['step']

    def __iter__(self):
        while True:
            video_num = np.random.choice(list(self.videos.keys()))
            video_gen = iter(self.videos[video_num])
            yield next(video_gen)

    def __getitem__(self, i):
        return self.videos[i][:]

def create(video_nums, **kwargs):
    os.makedirs('{}data'.format(WORKING_PATH), exist_ok=True)

    for video_num in video_nums:
        video_data = VideoData(video_num)
        video_data.process(**kwargs)

def clear():
    import shutil
    shutil.rmtree(WORKING_PATH + 'data')
