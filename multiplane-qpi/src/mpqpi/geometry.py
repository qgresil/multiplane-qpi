"""Panels (the 8 image fields on the two cameras) and the plane-to-reference affine warp (SI §6.4: affine + bilinear)."""
import numpy as np
from scipy import ndimage

ORIENT = {"none": lambda a: a, "lr": lambda a: a[..., ::-1], "ud": lambda a: a[..., ::-1, :], "lrud": lambda a: a[..., ::-1, ::-1]}


def panel(frames_by_stream, plane):
    """Cut one plane's field out of its camera stream and orient it like the reference.
    plane: calibration entry with "stream", "window_raw" {top, left, height, width} and "orientation"."""
    w = plane["window_raw"]
    a = frames_by_stream[plane["stream"]][..., w["top"]:w["top"] + w["height"], w["left"]:w["left"] + w["width"]]
    return ORIENT[plane["orientation"]](a)


def warp(img, A_from_ref, order=1, mode="constant", cval=np.nan):
    """Resample a panel image (2D, or a stack along axis 0) into reference coordinates.
    A_from_ref (2x3) maps reference (x, y, 1) -> panel (x, y); order=1 is the bilinear interpolation of SI §6.4.
    Outside the panel: NaN by default (mode="constant"), so that unlit / missing regions are never mistaken for data."""
    A = np.asarray(A_from_ref, float)
    mat = np.array([[A[1, 1], A[1, 0]], [A[0, 1], A[0, 0]]])
    off = np.array([A[1, 2], A[0, 2]])
    if img.ndim == 2:
        return ndimage.affine_transform(np.asarray(img, np.float32), mat, offset=off, order=order, mode=mode, cval=cval)
    return np.stack([warp(f, A, order, mode, cval) for f in img])


def fourier_shift(img, shift_yx):
    """Exact sub-pixel translation of a 2D image (periodic), used where interpolation blur must be avoided."""
    return np.real(np.fft.ifft2(ndimage.fourier_shift(np.fft.fft2(img), shift_yx)))
