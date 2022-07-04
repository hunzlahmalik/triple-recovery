from typing import NamedTuple
import numpy as np
import cv2
import time
from triplerecovery import blocks, authenticate, bits
from triplerecovery.constants import LOOKUPS


class RecoveryResult(NamedTuple):
    imarr: np.ndarray  # recovered array
    time: float


def _recreate(imarr: np.ndarray, recovery_bits: np.ndarray,
              tempred: np.ndarray, lookup: np.ndarray) -> np.ndarray:
    '''
    recreates the image using the recovery bit
    '''

    zoomx = 4  # according to the paper the zoom factor is 4
    average_bits_imarr = cv2.resize(imarr, None, fx=1/zoomx, fy=1 /
                                    zoomx, interpolation=cv2.INTER_AREA)
    average_bits_imarr = blocks.make(
        average_bits_imarr, (average_bits_imarr.shape[0]//zoomx, average_bits_imarr.shape[1]//zoomx), addChannel=False).reshape(16, -1)
    average_bits_imarr = np.unpackbits(average_bits_imarr, axis=1)

    # filling from the recovery bits
    average_bits = np.zeros(
        (recovery_bits.shape[0], recovery_bits.shape[1]//3), dtype=np.uint8)

    for partner in range(lookup.shape[0]):  # A,B,C,D
        for id in range(lookup.shape[1]):  # A1,A2,A3.....D4 etc
            if not np.any(tempred[lookup[partner, id]]):
                # this means that we can pick the averages from the average_bits_imarr
                # print("using average bits", partner, id)
                average_bits[lookup[partner, id]
                             ] = average_bits_imarr[lookup[partner, id]]
                continue

            # get the partner block of this id also the id it sits
            partner_block = -1, -1
            for i in range(lookup[partner].shape[0]):
                # if this id isn't tempred, we can stop
                if i != id and not np.any(tempred[lookup[partner, i]]):
                    partner_block = partner, i
                    break

            if partner_block == (-1, -1):
                print(
                    "Could not find partner block for partner", partner, "id", id, "=", lookup[partner, id])
                continue

            # get recovery bits of this id
            # get index of this id in the partner block
            idx = id
            if partner_block[1] <= id:
                idx -= 1

            # print("partner", partner, "id", id, "=", lookup[partner, id], "partner_block", partner_block, "idx",
            #       idx, "rshape", recovery_bits.shape, "range", idx*recovery_bits.shape[1]//3, idx*recovery_bits.shape[1]//3+recovery_bits.shape[1]//3)

            # now we have partner_block, so we can recover it
            # get the recovery bits of this partner block
            average_bits[lookup[partner, id]] = recovery_bits[lookup[partner_block],
                                                              idx*recovery_bits.shape[1]//3:idx*recovery_bits.shape[1]//3+recovery_bits.shape[1]//3]

    # now we have the average bits, so we can recreate the averages and then image

    # back to decimals
    averages = np.packbits(average_bits, axis=1)
    return blocks.combine(averages.reshape(averages.shape[0], imarr.shape[0]//16, imarr.shape[1]//16), (imarr.shape[0]//zoomx, imarr.shape[1]//zoomx), (imarr.shape[0]//16, imarr.shape[1]//16), channel=False)


def _recover(imarr: np.ndarray, recovery_bits: np.ndarray,
             tempred: np.ndarray, maskarr: np.ndarray, lookup: np.ndarray, interpolation: int) -> RecoveryResult:

    start_t = time.time()

    recoveredarr = imarr.copy()

    # only do if any of the blocks are tempred
    if np.any(tempred):
        smallim = _recreate(imarr, recovery_bits, tempred, lookup)

        # interpolate the small image, using CUBIC or LANCZOS4 preferably
        recoveredim = cv2.resize(smallim, imarr.shape[::-1],
                                 interpolation=interpolation)

        # now we have image, mask and recoveredimage
        # all we have to replace all the pixels in the image with the recovered image which are tempred
        recoveredarr[maskarr == 0] = recoveredim[maskarr == 0]

    return RecoveryResult(recoveredarr, time.time() - start_t)


def recover(imarr: np.ndarray, lookupidx: np.uint8 = 0, interpolation: int = cv2.INTER_LANCZOS4, key: str = "") -> RecoveryResult:
    if imarr.ndim > 3 or imarr.ndim < 2:
        raise Exception("Image array must be 3D or 2D!")

    lookup = LOOKUPS[lookupidx]

    # GREY
    if imarr.ndim == 2:
        # extracting the recovery bits
        recovery_bits = bits.recovery.extract(imarr, key)
        # extracting the auth bits
        auth = authenticate(imarr)

        # calling the recovery function
        return _recover(imarr, recovery_bits, auth.tempred, auth.maskarr, lookup, interpolation)

    # RGB
    if imarr.ndim == 3:
        start_t = time.time()

        retimarr = imarr.copy()

        for i in range(retimarr.shape[2]):
            # extracting the recovery bits
            recovery_bits = bits.recovery.extract(retimarr[:, :, i], key)
            # extracting the auth bits
            auth = authenticate(retimarr[:, :, i])

            # calling the recovery function
            retimarr[:, :, i] = _recover(
                retimarr[:, :, i], recovery_bits, auth.tempred, auth.maskarr, lookup, interpolation).imarr

        return RecoveryResult(retimarr, time.time() - start_t)
