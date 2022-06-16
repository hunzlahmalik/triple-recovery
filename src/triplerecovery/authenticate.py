from typing import NamedTuple
import numpy as np
import time
from triplerecovery.bits import authentication as bitsauth


class AuthenticationResult(NamedTuple):
    tempred: np.ndarray  # tempred array
    time: float


def authenticate(imarr: np.ndarray) -> AuthenticationResult:
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
    tempred = ~tempred

    return AuthenticationResult(tempred, time.time() - start_t)
