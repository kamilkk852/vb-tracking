import cv2
import numpy as np
import pickle
import json
import xml.etree.ElementTree as ET

SETTINGS = json.load(open('train/settings.json', 'r'))
DIFFS_STD = SETTINGS['DIFFS_STD']
MAX_SMALL_BREAK = 3
ACC_DIFF_THRESHOLDS = [20, 45, 80]

def read_kva_files(kva_paths, add_diameter=False):
    '''
    Reads Kinovea files and returns tagged frames numbers and corresponding ball
    locations.

    Parameters
    ----------
    kva_paths: list of strings

    add_diamater: bool, optional, default=False
        If True add column of diameters to the locations array.

    Returns
    -------
    frame_nums, locs: ndarray
    '''
    locs = []
    frame_nums = []

    for kva_path in kva_paths:
        root = ET.parse(kva_path).getroot()
        keyframes = root.find('Keyframes')

        for keyframe in keyframes.iter('Keyframe'):
            position = keyframe.find('Position')
            frame_num = int(position.attrib['UserTime'])

            drawings = keyframe.find('Drawings')

            if drawings is None:
                continue

            for drawing in drawings.iter('Drawing'):
                if drawing.attrib['Type'] != 'DrawingLine2D': #We process only Kinovea lines
                    continue

                #For having numpy (x - vertical, y - horizontal) nomenclature
                #we need to reverse the order
                y_start, x_start = map(int, drawing.find('m_StartPoint').text.split(';'))
                y_end, x_end = map(int, drawing.find('m_EndPoint').text.split(';'))

                loc_x = (x_start + x_end)/2
                loc_y = (y_start + y_end)/2

                if add_diameter:
                    diameter = np.sqrt((x_end - x_start)**2 + (y_end - y_start)**2)
                    locs.append([loc_x, loc_y, diameter])
                else: locs.append([loc_x, loc_y])

                frame_nums.append(frame_num - 1)

    #We want to have frame_nums sorted, because it makes frames reading faster
    frame_nums, locs = map(np.array, (frame_nums, locs))
    frame_nums_argsort = np.argsort(frame_nums)
    frame_nums, locs = map(lambda x: x[frame_nums_argsort], (frame_nums, locs))

    return frame_nums, locs

def process_frame(frame, target_size):
    '''
    Converts to RGB, resizes and pads a frame.

    Arguments
    ---------
    frame: ndarray, 3-D

    target_size: int

    Returns
    -------
    processed_frame: ndarray, 3-D
    '''
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.uint8)

    video_shape = np.array(frame.shape[:2])
    max_dim = np.max(video_shape)
    resized_shape = (video_shape * target_size / max_dim).astype(int)

    #We reverse resized_shape because the argument requires (width, height)
    resized_frame = cv2.resize(frame, tuple(reversed(resized_shape)))

    padding = ((0, target_size - resized_shape[0]),
               (0, target_size - resized_shape[1]),
               (0, 0))
    padded_frame = np.pad(resized_frame, padding,
                          mode='constant', constant_values=0)

    return padded_frame

def get_frames(video_path, frame_nums, target_size, print_breaks=False):
    '''
    Reads a video file and returns selected frames in form of numpy array.

    Arguments
    ---------
    video_path: str

    frame_nums: list or 1-D ndarray

    target_size: int
        Target frame size (width and height).
    print_breaks: bool, optional, default=False
        If True, notifies about frames with missing data.

    Returns
    -------
    frames: ndarray, 4-D
    '''
    cap = cv2.VideoCapture(video_path)

    frames = np.zeros((len(frame_nums), target_size, target_size, 3), dtype=np.uint8)

    for i, current_num in enumerate(frame_nums):
        previous_num = frame_nums[i-1]
        step = current_num - previous_num

        #If we're not reading consecutive frame, we need to set new position
        if step != 1:
            cap.set(1, current_num)

        if print_breaks and i != 0 and step <= MAX_SMALL_BREAK:
            print("Small break at: {}->{}".format(previous_num + 1, current_num + 1))

        ret, frame = cap.read()

        if ret:
            frames[i] = process_frame(frame, target_size)
        else: print('Could not load frame {}.'.format(i+1))

    return frames

def calc_diffs(frames, step, n_diffs):
    '''
    Calculates differences between frames.

    Arguments
    ---------
    frames: ndarray, 4-D

    step: int, more than 0
        Minimal step to be used for difference calculation.
    n_diffs: int, more than 0
        Number of differences (each with different step).

    Returns
    ------
    diffs: list of 4-D ndarrays
    '''
    diffs = []

    for n in range(n_diffs):
        diff_step = step + n
        diff = (frames[diff_step:].astype(np.float32) - \
                frames[:-diff_step].astype(np.float32)) / DIFFS_STD
        zero_fill = np.zeros((diff_step, frames.shape[1],
                              frames.shape[2], frames.shape[3]),
                             dtype=np.float32)
        diff_plus = np.concatenate([diff, zero_fill], axis=0)
        diff_minus = np.concatenate([zero_fill, -diff], axis=0)

        diffs += [diff_plus, diff_minus]

    return diffs

def calc_acc_diff(frames, thres, acc_size=10):
    '''
    Calculates accumulated frame difference.

    Arguments
    ---------
    frames: ndarray, 4-D

    thres: int
        Intensity change threshold - parameter of the algorithm.
    acc_size: int, optional, default=10
        Number of frames to accumulate over.
    '''
    intensity = np.mean(frames, axis=3, keepdims=True, dtype=np.float32)
    acc_diff = np.ones(intensity.shape, dtype=np.float32)

    for i in range(1, intensity.shape[0]):
        acc_diff[i] = np.maximum(0, acc_diff[i-1] - 1. / acc_size)
        acc_diff[i][np.abs(intensity[i] - intensity[i-1]) > thres] = 1

    return acc_diff

def get_input(frames, step, n_diffs):
    '''
    Performs transformation from frames data to the CNN input.

    Arguments
    ---------
    frames: ndarray, 4-D

    step: int, greater than 0

    n_diffs: int, greater than 0

    Returns
    -------
    input: ndarray, 4-D
    '''
    diffs = calc_diffs(frames, step, n_diffs)
    acc_diffs = [calc_acc_diff(frames, thres) for thres in ACC_DIFF_THRESHOLDS]
    frames = [frames.astype(np.float32) / 255]

    X = np.concatenate(frames + acc_diffs + diffs, axis=3)

    return X

def get_output(locs, n_boxes):
    '''
    Performs transformation from locations data to the CNN output.

    Arguments
    ---------
    locs: ndarray, 2-D

    n_boxes: int

    Returns
    -------
    output: ndarray, 3-D
    '''
    i = np.arange(locs.shape[0])
    x = locs[:, 0]
    y = locs[:, 1]

    box_x, box_y = map(lambda x: (x*n_boxes).astype(int), (x, y))

    y = np.zeros((locs.shape[0], n_boxes, n_boxes))
    y[i, box_x, box_y] = 1

    return y

def dump_chunks(X, y, path, chunk_size):
    '''
    Dumps data in chunks - each in different file.

    Arguments
    ---------
    X: ndarray, 4-D

    y: ndarray, 3-D

    path: str
        Saving path without chunk number and extension e.g. "E:/data0014_".
    chunk_size: int

    Returns
    -------
    None
    '''
    chunk_cnt = int(np.ceil(X.shape[0] / chunk_size))

    indices = np.arange(X.shape[0])
    index_chunks = np.array_split(indices, chunk_cnt)

    for i, index_chunk in enumerate(index_chunks):
        chunk_num_str = str(i).zfill(6)
        saving_path = '{}{}.h5'.format(path, chunk_num_str)

        file_stream = open(saving_path, 'wb')
        pickle.dump((X[index_chunk], y[index_chunk]), file_stream)

def plane_to_loc(plane):
    '''
    Transforms planes (CNN output) to estimated ball locations
    by weighting box positions by probability scores.

    Arguments
    ---------
    plane: ndarray, 3-D

    Returns
    -------
    locs: ndarray, 2-D
        Locations in (n_samples, 2) shape array form.
    '''
    n = plane.shape[1]

    #Add 0.5/n to get positions of box centers
    box_positions = np.linspace(0, 1, num=n, endpoint=False) + 0.5/n
    y_pos, x_pos = np.meshgrid(box_positions, box_positions)

    weighted_avgs = lambda vals, weights: np.sum((vals*weights).reshape(-1, n**2), axis=1) \
                                            / np.sum(weights.reshape(-1, n**2), axis=1)

    locs = np.hstack([weighted_avgs(pos, plane).reshape(-1, 1) for pos in [x_pos, y_pos]])

    return locs
