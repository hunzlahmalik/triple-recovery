from typing import NamedTuple
import numpy as np
import time
import triplerecovery.bits as bits

CHECK_EMBEDDING = False


class EmbeddingResult(NamedTuple):
    imarr: np.ndarray
    time: float
    is_checked: bool = False
    is_correct: bool = False


def _embed(imarr: np.ndarray, lookup: np.ndarray | None = None) -> EmbeddingResult:
    """
    Embed watermark into image.
    @Returns: dict[np.ndarray, int]
    embedded image, time taken to embed
    """

    if lookup is None:
        lookup = np.array([
            [0, 7, 13, 10],
            [1, 6, 12, 11],
            [4, 2, 9, 15],
            [5, 3, 8, 14]], dtype=np.uint8)

    if not CHECK_EMBEDDING:
        # make embedding and just return
        start_t = time.time()
        recvim = bits.recovery.embed(
            imarr, bits.recovery.make(imarr, lookup))
        return EmbeddingResult(bits.authentication.embed(
            recvim,
            bits.authentication.make(recvim)
        ),  time.time() - start_t)

    # make embedding and also check if it's correct
    start_t = time.time()
    # make recoverybits
    recovery_bits = bits.recovery.make(imarr, lookup)
    recvim = bits.recovery.embed(imarr, recovery_bits)

    # make hashes
    hashes = bits.authentication.make(recvim)

    embeddedim = bits.authentication.embed(
        recvim,
        hashes
    )

    # extract recoverybits
    exrecovery = bits.recovery.extract(embeddedim)

    # extract hashes
    exhashes = bits.authentication.extract(embeddedim)

    return EmbeddingResult(embeddedim, time.time() - start_t, True, np.array_equal(recovery_bits, exrecovery) and np.array_equal(hashes, exhashes))


def embed(imarr: np.ndarray, lookup: np.ndarray | None = None) -> EmbeddingResult:
    if imarr.ndim > 3 or imarr.ndim < 2:
        raise Exception("Image array must be 3D or 2D!")

    # GREY
    if imarr.ndim == 2:
        return _embed(imarr, lookup)

    # RGB
    if imarr.ndim == 3:
        start_t = time.time()

        retimarr = imarr.copy()

        for i in range(imarr.shape[2]):
            retimarr[:, :, i] = _embed(retimarr[:, :, i], lookup).imarr

        return EmbeddingResult(retimarr, time.time() - start_t)
