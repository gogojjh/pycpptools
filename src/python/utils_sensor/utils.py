import numpy as np

def correct_intrinsic_scale(K, scale_x, scale_y):
    """Given an intrinsic matrix (3x3) and two scale factors, returns the new intrinsic matrix corresponding to
    the new coordinates x' = scale_x * x; y' = scale_y * y
    Source: https://dsp.stackexchange.com/questions/6055/how-does-resizing-an-image-affect-the-intrinsic-camera-matrix
    """

    transform = np.eye(3)
    transform[0, 0] = scale_x
    transform[0, 2] = scale_x / 2 - 0.5
    transform[1, 1] = scale_y
    transform[1, 2] = scale_y / 2 - 0.5
    Kprime = transform @ K

    return Kprime
