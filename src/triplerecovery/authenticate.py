from typing import NamedTuple
import cv2
import numpy as np
import time
from triplerecovery import blocks
from triplerecovery.bits import authentication as bitsauth


class AuthenticationResult(NamedTuple):
    tempred: np.ndarray  # tempred array
    maskarr: np.ndarray  # mask array
    time: float


def _authenticate(imarr: np.ndarray) -> AuthenticationResult:
    if imarr.ndim != 2:
        raise Exception(
            "Image array must be 2D! Given dims {}".format(imarr.ndim))

    start_t = time.time()

    # make hashes
    hashes = bitsauth.make(imarr)

    # extract hashes
    exhashes = bitsauth.extract(imarr)

    tempred = np.zeros((exhashes.shape[0], exhashes.shape[1]), dtype=bool)

    for i in range(exhashes.shape[0]):
        for j in range(exhashes.shape[1]):
            tempred[i, j] = np.array_equal(hashes[i, j], exhashes[i, j])

    maskarr = cv2.resize(blocks.combine(np.where(tempred.astype(np.uint8) == 1, 255, tempred.astype(
        np.uint8)), (imarr.shape[0]//16, imarr.shape[1]//16), (imarr.shape[0]//64, imarr.shape[1]//64), channel=False), (imarr.shape[1], imarr.shape[0]), interpolation=cv2.INTER_NEAREST)

    tempred = ~tempred

    return AuthenticationResult(tempred, maskarr, time.time() - start_t)


def authenticate(imarr: np.ndarray) -> AuthenticationResult:
    # GREY
    if imarr.ndim == 2:
        # calling the recovery function
        return _authenticate(imarr)

    # RGB
    if imarr.ndim == 3:
        start_t = time.time()

        retmask = imarr.copy()

        for i in range(retmask.shape[2]):
            # calling the recovery function
            temp = _authenticate(retmask[:, :, i])
            retmask[:, :, i] = temp.maskarr

        return AuthenticationResult(None, retmask, time.time() - start_t)
