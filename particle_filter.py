import numpy as np

STEP_SIZE = 6
N_PARTICLES = 100

def correct_pred(acc_diff, pred_pos, window=30):
    x_loc, y_loc = pred_pos
    start_x = int(max(0, x_loc - window))
    start_y = int(max(0, y_loc - window))
    end_x = int(min(acc_diff.shape[0], x_loc + window))
    end_y = int(min(acc_diff.shape[1], y_loc + window))

    y, x = np.meshgrid(np.arange(end_x - start_x), np.arange(end_x - start_x))

    dis_x = x - (x_loc - start_x)
    dis_y = y - (y_loc - start_y)

    acc_diff = acc_diff[start_x:end_x, start_y:end_y]

    brightest = acc_diff*(acc_diff == np.max(acc_diff))

    dis = dis_x**2 + dis_y**2
    dis *= brightest

    pred_loc = np.argwhere(dis == np.min(dis[dis > 0])).mean(axis=0)
    pred_loc += np.array([start_x, start_y])

    return pred_loc

def resample(weights):
    n = len(weights)
    indices = []
    C = [0.] + [sum(weights[:i+1]) for i in range(n)]
    u0, j = np.random.random(), 0
    for u in [(u0+i)/n for i in range(n)]:
        while u > C[j]:
            j += 1
        indices.append(j-1)
    return indices


def predict(pos, sequence, **kwargs):
    step_size = kwargs.get('step_size', STEP_SIZE)
    n_particles = kwargs.get('n_particles', N_PARTICLES)
    reverse = kwargs.get('reverse', False)

    preds = [pos]

    if reverse:
        seq = iter(reversed(sequence))
    else: seq = iter(sequence)

    x = np.ones((n_particles, 2), int) * pos                   # Initial position
    f0 = next(seq)[tuple(pos)] * np.ones(n_particles)         # Target colour model

    for im in seq:
        np.add(x, np.random.uniform(-step_size, step_size, x.shape),
               out=x, casting="unsafe")  # Particle motion model: uniform step
        x = x.clip(np.zeros(2), np.array(im.shape)-1).astype(int) # Clip out-of-bounds particles
        f = im[tuple(x.T)]                         # Measure particle colours
        w = 1./(1. + (f0-f)**2)                    # Weight~ inverse quadratic colour distance
        w /= sum(w)                                 # Normalize w
        preds.append(np.sum(x.T*w, axis=1))     # Return expected position, particles and weights
        if 1./sum(w**2) < n_particles/2.:                     # If particle cloud degenerate:
            x = x[resample(w), :]

    if reverse:
        return list(reversed(preds))
    return preds
