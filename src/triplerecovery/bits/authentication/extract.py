import numpy as np
import hashlib
from triplerecovery.blocks import make as make_blocks
from triplerecovery.utils import get_bit
from triplerecovery.constants import HASH_SIZE

'''
Will make the recovery bits for image
'''

'''
Data is one 16x16 block converted into four 8x8 blocks
'''


def get_bits(data: np.ndarray) -> np.ndarray:
    '''
    gets data from last two lsb of data
    '''
    if data.shape != (8, 8):
        print(f"Warning! given size {data.shape} instead of (8, 8)")
    local_data = data.copy()
    bits = np.zeros((64, 2), np.uint8)
    bits[:, 0] = np.fromiter((get_bit(d, 0)
                             for d in local_data.flat), dtype=np.uint8)
    bits[:, 1] = np.fromiter((get_bit(d, 1)
                             for d in local_data.flat), dtype=np.uint8)
    return bits


def extractfrom8x8(b8x8: np.ndarray) -> np.ndarray:
    # creating hash space
    # reshaping it into 2bits in last for better placement
    hashes = np.zeros(
        (*b8x8[:, 0].shape[:2], (HASH_SIZE*8)//2, 2), dtype=object)

    # extracting
    for i in range(b8x8[:, 0].shape[0]):
        for j in range(b8x8[:, 0].shape[1]):
            hashes[i][j] = get_bits(b8x8[i, 0, j, 3, :, :])

    return hashes.reshape(hashes.shape[0], hashes.shape[1], hashes.shape[2]*2)


def extract(imarr: np.ndarray) -> np.ndarray:
    '''
    '''

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

    hashes = extractfrom8x8(b8x8)

    ################################################################################

    return hashes
