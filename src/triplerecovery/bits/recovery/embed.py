import numpy as np
import triplerecovery.blocks as blocks
from triplerecovery.utils import set_bit

'''
Will embed the recovery bits in image
'''


def _embedfrom8x8(b8x8: np.ndarray, recovery_bits: np.ndarray) -> np.ndarray:
    # Now put the recovery bit in the first and second LSB of blocks4x4

    # changing the shape of recovery bit to match the shape of b8x8
    # we are given a space of two bit, so we need to make shape (16, ..., 2)
    bits = recovery_bits.reshape(
        recovery_bits.shape[0], int(recovery_bits.shape[1]//2), 2)

    # setting the bits in the blocks
    # using first 3 8x8blocks
    for i in range(b8x8[:, 0].shape[0]):
        for j in range(len(b8x8[i, 0, :, :3, :, :].flat)):
            number = b8x8[i, 0, :, :3, :, :].flat[j]
            b8x8[i, 0, :, :3, :, :].flat[j] = set_bit(
                set_bit(number, 0, bits[i][j][0]), 1, bits[i][j][1])

    return b8x8


def embed(imarr: np.ndarray, recovery_bits: np.ndarray) -> np.ndarray:
    '''
    '''

    # Recovery bits creation completed
    # Now we need to make the space to put these recovery bits

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

    # Making 8x8 blocks of those 16x16 Step 9
    b8x8_shape = (8, 8)
    # reshaping because we needed that shape
    b8x8 = b16x16.reshape(
        *b16x16.shape[:-2],
        int((b16x16_shape[0]*b16x16_shape[1]) /
            (b8x8_shape[0]*b8x8_shape[1])),
        b8x8_shape[0], b8x8_shape[1])

    for i in range(b16x16.shape[0]):
        for j in range(b16x16.shape[2]):
            b8x8[i][0][j] = blocks.make(b16x16[i][0][j].copy(), b8x8_shape,
                                        addChannel=False)

    ################################################################################

    _embedfrom8x8(b8x8, recovery_bits)

    ################################################################################

    # Now to combine the blocks back to Image
    c16x16 = b16x16.copy()
    for i in range(c16x16.shape[0]):
        for j in range(c16x16.shape[2]):
            c16x16[i][0][j] = blocks.combine(b8x8[i][0][j].copy(), imageshape=(
                16, 16), blockshape=b8x8_shape, channel=False)

    # merging 16x16
    cmainblocks = mainblocks.copy()
    for i in range(cmainblocks.shape[0]):
        cmainblocks[i][0] = blocks.combine(
            c16x16[i][0], mainblock_shape, blockshape=b16x16_shape, channel=False)

    # merging main blocks to main image
    return blocks.combine(cmainblocks.copy(), imarr.shape,
                          mainblock_shape).reshape(imarr.shape)
