import math

import numpy as np


def fibonacci_sphere(n_samples):
    def populate(idx):
        offset = 2.0 / n_samples
        increment = math.pi * (3.0 - math.sqrt(5.0))
        y = ((idx * offset) - 1) + (offset / 2.0)
        r = math.sqrt(1 - y ** 2)
        phi = (idx % n_samples) * increment

        x = math.cos(phi) * r
        z = math.sin(phi) * r

        return x, y, z

    g = np.vectorize(populate)
    return np.fromfunction(g, (n_samples,))

def fib_plane(n_samples):
    r_start = 0.0
    r_end = 1.0
    part_of_circle = 1
    r_step = (r_end - r_start) / n_samples
    golden_angle = math.pi * (3.0 - math.sqrt(5.0)) * part_of_circle

    def populate(idx):
        r = np.sqrt(r_start + r_step * idx)

        phi = (idx * golden_angle) % (part_of_circle * 2 * math.pi)
        y = math.cos(phi) * r
        x = math.sin(phi) * r
        return x, y

    g = np.vectorize(populate)
    return np.fromfunction(g, (n_samples,))
