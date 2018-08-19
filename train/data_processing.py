import cv2
import numpy as np
import pickle
import xml.etree.ElementTree as ET

DIFFS_STD = 6

def read_kva_files(kva_paths, add_diameter=False):
    locs = []
    frame_nums = []

    for kva_path in kva_paths:
        root = ET.parse(kva_path).getroot()
        keyframes = root.find('Keyframes')

        for keyframe in keyframes.iter('Keyframe'):
            position = keyframe.find('Position')
            frame_num = int(position.attrib['UserTime'])

            drawings = keyframe.find('Drawings')

            if drawings is not None:
                for drawing in drawings.iter('Drawing'):
                    if drawing.attrib['Type'] != 'DrawingLine2D':
                        continue

                    y_start, x_start = map(int, drawing.find('m_StartPoint').text.split(';'))
                    y_end, x_end = map(int, drawing.find('m_EndPoint').text.split(';'))

                    loc_x = (x_start + x_end)/2
                    loc_y = (y_start + y_end)/2

                    if add_diameter:
                        diameter = np.sqrt((x_end - x_start)**2 + (y_end - y_start)**2)
                        locs.append([loc_x, loc_y, diameter])
                    else: locs.append([loc_x, loc_y])

                    frame_nums.append(frame_num - 1)

    frame_nums, locs = map(np.array, (frame_nums, locs))
    frame_nums_argsort = np.argsort(frame_nums)
    frame_nums, locs = map(lambda x: x[frame_nums_argsort], (frame_nums, locs))

    return frame_nums, locs

def get_frames(video_path, frame_nums, input_size, print_breaks=False):
    cap = cv2.VideoCapture(video_path)

    frames = []

    def resize_and_pad(frame):
        video_shape = np.array(frame.shape[:2])
        max_dim = np.max(video_shape)
        resized_shape = (video_shape * input_size / max_dim).astype(int)

        resized_frame = cv2.resize(frame, tuple(reversed(resized_shape)))

        padding = ((0, input_size - resized_shape[0]),
                   (0, input_size - resized_shape[1]),
                   (0, 0))
        padded_frame = np.pad(resized_frame, padding,
                              mode='constant', constant_values=0)

        return padded_frame

    previous_num = 0

    for i in range(len(frame_nums)):
        if i > 0:
            previous_num = frame_nums[i-1]

        current_num = frame_nums[i]

        if current_num - previous_num != 1:
            cap.set(1, current_num)
            if print_breaks and current_num != 0 and current_num - previous_num <= 3:
                print("Small break at: {}->{}".format(previous_num + 1, current_num + 1))
        ret, frame = cap.read()

        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.uint8)
            frame = resize_and_pad(frame)
            frames.append(frame)
        else: print('Could not load frame {}.'.format(i+1))

    assert frames
    frames = np.array(frames)

    return frames

def calc_diffs(frames, step, n_diffs):
    diffs = []

    for n in range(n_diffs):
        diff_step = step + n
        diff = (frames[diff_step:].astype(np.float32) - \
                frames[:-diff_step].astype(np.float32)) / DIFFS_STD
        zero_fill = np.zeros((diff_step, frames.shape[1],
                              frames.shape[2], frames.shape[3]))
        diff_plus = np.concatenate([diff, zero_fill], axis=0)
        diff_minus = np.concatenate([zero_fill, -diff], axis=0)

        diffs += [diff_plus, diff_minus]

    return diffs

def calc_acc_diff(frames, thres, frames_cnt=10):
    amax = frames_cnt
    intensity = np.mean(frames, axis=3)
    acc_diff = np.zeros(intensity.shape)
    norm_acc_diff = np.zeros(intensity.shape)

    for i in range(intensity.shape[0]):
        if i == 0:
            acc_diff[i] = amax
        else:
            acc_diff[i] = acc_diff[i-1]-1
            acc_diff[i][np.abs(intensity[i]-intensity[i-1]) > thres] = amax

        norm_acc_diff[i] = np.clip(acc_diff[i], 0, frames_cnt)/frames_cnt

    return np.expand_dims(norm_acc_diff, axis=3)

def get_input(frames, step, n_diffs):
    diffs = calc_diffs(frames, step, n_diffs)
    acc_diffs = [calc_acc_diff(frames, thres) for thres in [20, 45, 80]]
    frames = [frames.astype(np.float32) / 255]

    X = np.concatenate(frames + acc_diffs + diffs, axis=3)

    return X

def get_output(locs, n_boxes):
    i = np.arange(locs.shape[0])
    x = locs[:, 0]
    y = locs[:, 1]

    box_x, box_y = map(lambda x: (x*n_boxes).astype(int), (x, y))

    y = np.zeros((locs.shape[0], n_boxes, n_boxes))
    y[i, box_x, box_y] = 1

    return y

def dump_chunks(X, y, path, chunk_size):
    chunk_cnt = int(np.ceil(X.shape[0] / chunk_size))

    indices = np.arange(X.shape[0])
    index_chunks = np.array_split(indices, chunk_cnt)

    for i, index_chunk in enumerate(index_chunks):
        chunk_num_str = str(i).zfill(6)
        saving_path = '{}{}.h5'.format(path, chunk_num_str)

        file_stream = open(saving_path, 'wb')
        pickle.dump((X[index_chunk], y[index_chunk]), file_stream)

def plane_to_loc(plane):
    n = plane.shape[1]

    box_positions = np.linspace(0, 1, num=n, endpoint=False) + 0.5/n
    y_pos, x_pos = np.meshgrid(box_positions, box_positions)

    weighted_avgs = lambda vals, weights: np.sum((vals*weights).reshape(-1, n**2), axis=1) \
                                            / np.sum(weights.reshape(-1, n**2), axis=1)

    locs = np.hstack([weighted_avgs(pos, plane).reshape(-1, 1) for pos in [x_pos, y_pos]])

    return locs
