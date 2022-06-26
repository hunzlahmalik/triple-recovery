from typing import NamedTuple
import numpy as np
import cv2
import time
from triplerecovery import blocks, authenticate, bits


class RecoveryResult(NamedTuple):
    imarr: np.ndarray  # recovered array
    time: float


def _recover(imarr: np.ndarray, recovery_bits: np.ndarray,
             tempred: np.ndarray, lookup: np.ndarray, interpolation: int, key: str) -> RecoveryResult:

    if lookup is None:
        lookup = np.array([
            [0, 7, 13, 10],
            [1, 6, 12, 11],
            [4, 2, 9, 15],
            [5, 3, 8, 14]], dtype=np.uint8)

    start_t = time.time()

    recoveredarr = imarr.copy()

    # only do if any of the blocks are tempred
    if np.any(tempred):

        # converting the image to 16x16 blocks
        # making 16 main blocks
        # size of single main block S= M/sqrt(16) X T=N/sqrt(16)
        # e.g for lena the 512x512 the partner blocks size would be 16 blocks each with size (512x512)/(4x4) = 128x128
        mainblock_shape = (int(imarr.shape[0]/4), int(imarr.shape[1]/4))
        mainblocks = blocks.make(imarr.copy(), mainblock_shape)

        # here the main blocks are like this index.
        # 1st index is the block number.
        # 2nd index is the channel (RGB) or 0 in Grey.
        # 3rd and 4th are for indexing the block.

        # Making 16x16 for Step 8
        # Dividing the main blocks to 16x16 blocks
        # Total blocks = SxT/16x16 = 128x128/16x16 = 64 Blocks

        b16x16_shape = (16, 16)
        # reshaping because we needed that shape
        b16x16 = mainblocks.reshape(
            *mainblocks.shape[:-2],
            int((mainblock_shape[0]*mainblock_shape[1]) /
                (b16x16_shape[0]*b16x16_shape[1])),
            b16x16_shape[0], b16x16_shape[1]).copy()

        for i in range(mainblocks.shape[0]):
            b16x16[i][0] = blocks.make(mainblocks[i][0], b16x16_shape,
                                       addChannel=False)

        for partner in range(lookup.shape[0]):  # A,B,C,D
            for id in range(lookup.shape[1]):  # A1,A2,A3.....D4 etc
                # check if this is tempred
                if np.any(tempred[lookup[partner, id]]):
                    # this mainblock is tempred, so we need to recover it
                    # but which 16x16 block is tempred?

                    # get the partner block of this id also the id it sits
                    partner_block = -1, -1
                    for i in range(lookup[partner].shape[0]):
                        # if this id isn't tempred, we can stop
                        if i != id and not np.any(tempred[lookup[partner, i]]):
                            partner_block = partner, i
                            break

                    if partner_block == (-1, -1):
                        print(
                            "Could not find partner block for {} {}".format(partner, id))
                        continue

                    # get recovery bits of this id
                    # get index of this id in the partner block
                    idx = id
                    if partner_block[1] < id:
                        idx -= 1

                    # now we have partner_block, so we can recover it
                    # get the recovery bits of this partner block
                    recovery_bits_partner = recovery_bits[lookup[partner_block],
                                                          idx*recovery_bits.shape[1]//3:idx*recovery_bits.shape[1]//3+recovery_bits.shape[1]//3]

                    # now we have recovrey bits the exact partner block
                    # but these are in binary, so we need to convert them to uint
                    recovery_decimals = np.packbits(recovery_bits_partner)

                    # these recovery bits are for all the main blocks 4x4
                    # but we are going to replace only the tempred 16x16

                    # shaping these recovery bits to the 16x16 blocks
                    _zoomshape = int(
                        imarr.shape[0]/(4*4)), int(imarr.shape[1]/(4*4))
                    r16x16 = blocks.make(cv2.resize(
                        recovery_decimals.reshape(
                            _zoomshape), (_zoomshape[0]*4, _zoomshape[1]*4),
                        interpolation=interpolation),
                        b16x16_shape, addChannel=False
                    )

                    # now we repace which ever 16x16 block is tempred
                    for i in range(tempred[lookup[partner, id]].shape[0]):
                        if tempred[lookup[partner, id], i]:
                            # means this 16x16 block is tempred
                            b16x16[lookup[partner, id], 0, i] = r16x16[i]

        # combining the 16x16 blocks to the main blocks
        # merging 16x16
        cmainblocks = mainblocks.copy()
        for i in range(cmainblocks.shape[0]):
            cmainblocks[i][0] = blocks.combine(
                b16x16[i][0], mainblock_shape, blockshape=b16x16_shape, channel=False)

        # merging main blocks to main image
        recoveredarr = blocks.combine(cmainblocks.copy(), imarr.shape,
                                      mainblock_shape).reshape(imarr.shape)

    return RecoveryResult(recoveredarr, time.time() - start_t)


def recover(imarr: np.ndarray, lookup: np.ndarray | None = None, interpolation: int = cv2.INTER_CUBIC, key: str = "key") -> RecoveryResult:
    if imarr.ndim > 3 or imarr.ndim < 2:
        raise Exception("Image array must be 3D or 2D!")

    # GREY
    if imarr.ndim == 2:
        # extracting the recovery bits
        recovery_bits = bits.recovery.extract(imarr, key)
        # extracting the auth bits
        auth_bits = authenticate(imarr).tempred

        # calling the recovery function
        return _recover(imarr, recovery_bits, auth_bits, lookup, interpolation, key)

    # RGB
    if imarr.ndim == 3:
        start_t = time.time()

        retimarr = imarr.copy()

        for i in range(retimarr.shape[2]):
            # extracting the recovery bits
            recovery_bits = bits.recovery.extract(retimarr[:, :, i], key)
            # extracting the auth bits
            auth_bits = authenticate(retimarr[:, :, i]).tempred

            # calling the recovery function
            retimarr[:, :, i] = _recover(
                retimarr[:, :, i], recovery_bits, auth_bits, lookup, interpolation, key).imarr

        return RecoveryResult(retimarr, time.time() - start_t)
