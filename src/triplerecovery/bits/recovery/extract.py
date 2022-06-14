import numpy as np
from triplerecovery.blocks import make as make_blocks
from triplerecovery.utils import get_bit

'''
Will extract the recovery bits in image
'''


def _extractfrom8x8(b8x8: np.ndarray, imarr_shape: tuple, avgblocks_shape: tuple = (4, 4)) -> np.ndarray:
    bits = np.zeros((16, (int(((imarr_shape[0]//4)*(imarr_shape[1]//4)) /
                              (avgblocks_shape[0]*avgblocks_shape[1]))*3*8) // 2, 2), dtype=np.uint8)

    for i in range(b8x8[:, 0].shape[0]):
        for j in range(len(b8x8[i, 0, :, :3, :, :].flat)):
            number = b8x8[i, 0, :, :3, :, :].flat[j]
            bits[i][j][0] = get_bit(number, 0)
            bits[i][j][1] = get_bit(number, 1)

    return bits.reshape(16, -1)


def extract(imarr: np.ndarray) -> np.ndarray:
    '''
    '''

    # Recovery bits creation completed
    # Now we need to make the space to put these recovery bits

    # making 16 main blocks
    # size of single main block S= M/sqrt(16) X T=N/sqrt(16)
    # e.g for lena the 512x512 the partner blocks size would be 16 blocks each with size (512x512)/(4x4) = 128x128
    mainblock_shape = (int(imarr.shape[0]/4), int(imarr.shape[1]/4))
    mainblocks = make_blocks(imarr.copy(), mainblock_shape)

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
        b16x16[i][0] = make_blocks(mainblocks[i][0], b16x16_shape,
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
            b8x8[i][0][j] = make_blocks(b16x16[i][0][j].copy(), b8x8_shape,
                                        addChannel=False)

    ################################################################################

    bits = _extractfrom8x8(b8x8, imarr.shape)

    ################################################################################

    return bits
