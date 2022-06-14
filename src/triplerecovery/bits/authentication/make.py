import numpy as np
import hashlib
from triplerecovery.blocks import make as make_blocks
from triplerecovery.utils import set_lsb_zero

'''
Will make the recovery bits for image
'''

'''
Data is one 16x16 block converted into four 8x8 blocks
'''

HASH_SIZE = 16


def hash_block(data: np.ndarray, key: str = None, digest_size=HASH_SIZE, extras=[]) -> hashlib:
    if data.shape != (4, 8, 8):
        print(f"Warning! given size {data.shape} instead of (4, 8, 8)")
    local = data.copy().astype(np.int8)  # copying to avoid overighting lsb
    local[-1] = set_lsb_zero(local[-1])  # setting last 8x8 blocks lsb zero
    if key is None:
        h = hashlib.blake2b(digest_size=digest_size)
    else:
        h = hashlib.blake2b(key=key.encode())
    h.update(local.data)
    for extra in extras:
        h.update(extra.encode())
    return h


def bin2np(binstr) -> np.ndarray:
    return np.frombuffer(binstr, dtype=np.uint8)


def makefrom8x8(b8x8: np.ndarray) -> np.ndarray:
    # creating hash
    hashes = np.zeros((*b8x8[:, 0].shape[:2], HASH_SIZE*8), dtype=object)
    for i in range(b8x8[:, 0].shape[0]):
        for j in range(b8x8[:, 0].shape[1]):
            hashes[i][j] = np.unpackbits(bin2np(hash_block(
                b8x8[i, 0, j]).digest()))
    return hashes


def make(imarr: np.ndarray) -> np.ndarray:
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

    hashes = makefrom8x8(b8x8)

    ################################################################################

    return hashes
