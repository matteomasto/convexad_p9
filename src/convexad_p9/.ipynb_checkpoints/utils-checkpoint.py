
import numpy as np
import scipy.fft
from scipy.ndimage import center_of_mass, uniform_filter
import tensorflow as tf
from tensorflow.signal import fft3d, ifft3d, fftshift, ifftshift

# =============================================================================
# CROPPING / RESAMPLING UTILITIES
# =============================================================================

def crop_center_3d(array: np.ndarray, side: int) -> np.ndarray:
    """
    Extract a centered cubic subvolume.

    Parameters
    ----------
    array : np.ndarray
        Input array of shape (D, H, W)
    side : int
        Side length of cubic crop

    Returns
    -------
    np.ndarray
        Cropped array of shape (side, side, side)
    """
    if side > min(array.shape):
        raise ValueError("Crop size exceeds array dimensions.")

    center = np.array(array.shape) // 2
    start = center - side // 2
    end = start + side

    return array[start[0]:end[0], start[1]:end[1], start[2]:end[2]]


def crop_tensor_half_size(tensor: np.ndarray) -> np.ndarray:
    """
    Crop spatial dimensions by factor 2 (centered).

    Assumes shape: (B, D, H, W)

    Returns
    -------
    np.ndarray
        Cropped tensor of shape (B, D/2, H/2, W/2)
    """
    _, D, H, W = tensor.shape
    return tensor[
        :,
        D//4:3*D//4,
        H//4:3*H//4,
        W//4:3*W//4
    ]


def crop_around_com(array: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    """
    Crop (or pad) around center of mass.

    Parameters
    ----------
    array : np.ndarray
        Input 3D array
    target_shape : tuple
        Desired output shape

    Returns
    -------
    np.ndarray
    """
    if len(target_shape) != 3:
        raise ValueError("target_shape must be length 3.")

    com = np.array(center_of_mass(array))
    target = np.array(target_shape)

    start = np.floor(com - target // 2).astype(int)
    end = start + target

    out = np.zeros(target_shape, dtype=array.dtype)

    input_slices = []
    output_slices = []

    for i in range(3):
        i0 = max(start[i], 0)
        i1 = min(end[i], array.shape[i])
        input_slices.append(slice(i0, i1))

        o0 = max(-start[i], 0)
        o1 = o0 + (i1 - i0)
        output_slices.append(slice(o0, o1))

    out[tuple(output_slices)] = array[tuple(input_slices)]
    return out


def rebin_3d(data: np.ndarray, bins: tuple[int, int, int]) -> np.ndarray:
    """
    Rebin by local averaging + subsampling.

    Parameters
    ----------
    data : np.ndarray
        Input 3D array
    bins : tuple
        (bx, by, bz)

    Returns
    -------
    np.ndarray
    """
    smoothed = uniform_filter(data, size=bins, mode="constant")
    return smoothed[::bins[0], ::bins[1], ::bins[2]]


def symmetric_pad_or_crop(
    data: np.ndarray,
    output_shape: tuple[int, ...],
    values: float = 0.0
) -> np.ndarray:
    """
    Symmetric padding or COM-centered cropping.

    Cropping is performed around COM of |data|.

    Returns
    -------
    np.ndarray
    """
    if data.ndim != len(output_shape):
        raise ValueError("Dim mismatch.")

    result = data
    data_shape = np.array(data.shape)

    for axis, (cur, out) in enumerate(zip(data_shape, output_shape)):

        if cur < out:
            pad_before = (out - cur) // 2
            pad_after = (out - cur) - pad_before

            pad = [(0, 0)] * data.ndim
            pad[axis] = (pad_before, pad_after)

            result = np.pad(result, pad, constant_values=values)

        elif cur > out:
            com = center_of_mass(np.abs(result))
            c = int(round(com[axis]))

            start = max(0, c - out // 2)
            end = min(start + out, cur)

            start = end - out  # ensure exact size

            sl = [slice(None)] * data.ndim
            sl[axis] = slice(start, end)

            result = result[tuple(sl)]

    return result


# =============================================================================
# FOURIER PROJECTION
# =============================================================================

def project(obj: np.ndarray, Iobs: np.ndarray) -> np.ndarray:
    """
    Fourier modulus projection.

    Enforces |F(obj)| = sqrt(Iobs).

    Returns
    -------
    np.ndarray (complex)
    """
    F = scipy.fft.ifftshift(scipy.fft.fftn(scipy.fft.fftshift(obj)))
    F = np.sqrt(Iobs) * np.exp(1j * np.angle(F))
    return scipy.fft.ifftshift(scipy.fft.ifftn(scipy.fft.fftshift(F)))

# =============================================================================
# IO
# =============================================================================

def save_model_npz(model, filename):
    np.savez(
        filename,
        **{v.name.replace(":0", ""): v.numpy() for v in model.trainable_variables}
    )


def load_model_npz(model, filename):
    data = np.load(filename)

    var_map = {v.name.replace(":0", ""): v for v in model.trainable_variables}

    for k in data.files:
        if k not in var_map:
            raise KeyError(f"{k} not in model.")

        var_map[k].assign(data[k])
