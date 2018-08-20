import numpy as np

STEP_SIZE = 6
N_PARTICLES = 100

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
