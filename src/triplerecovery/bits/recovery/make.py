import numpy as np
from triplerecovery.blocks import make as make_blocks
from triplerecovery.utils import shuffle_under_seed

'''
Will make the recovery bits for image
'''


def make(imarr: np.ndarray, lookup: np.ndarray, key: str) -> np.ndarray:
    '''
    '''

    # making 16 main blocks
    # size of single main block S= M/sqrt(16) X T=N/sqrt(16)
    # e.g for lena the 512x512 the partner blocks size would be 16 blocks each with size (512x512)/(4x4) = 128x128
    mainblock_shape = (int(imarr.shape[0]/4), int(imarr.shape[1]/4))
    mainblocks = make_blocks(imarr, mainblock_shape)

    # here the main blocks are like this index.
    # 1st index is the block number.
    # 2nd index is the channel (RGB) or 0 in Grey.
    # 3rd and 4th are for indexing the block.

    # Step3
    # Ab har main block ko divide karo k 4x4 k block ban jain
    # numberOfBlocks=(SxT)/(4x4)
    # e.g 128x128/4x4 = 1024

    avgblocks_shape = (4, 4)
    averages = np.zeros((16, int((mainblock_shape[0]*mainblock_shape[1]) /
                                 (avgblocks_shape[0]*avgblocks_shape[1]))), dtype=np.uint8)
    # 4 indicatior A,B,C,D, 4 blocks of A, then the 4x4 Blocks which have count = (SxT)/(4x4), e.g 1024

    for partner in lookup:  # A,B,C,D
        for id in partner:  # A1,A2,A3.....D4 etc
            averages[id] = make_blocks(mainblocks[id][0].copy(), avgblocks_shape,
                                       addChannel=False).mean((1, 2))

    # array([[162, 162, 162, 161],
    #       [162, 162, 162, 161],
    #       [162, 162, 162, 161],
    #       [162, 162, 162, 161]], dtype=uint8)
    # this will give you the avg 161. But it sould be 162 as it's dominant. Minor improvemnt required

    # now we have average of every mainblock according to 4x4. Which in total are 1024

    # time to convert them into the binary
    average_bits = np.unpackbits(averages, axis=1)

    # merging partner blocks average to make the recovery bits
    recovery_bits = np.zeros(
        (average_bits.shape[0], average_bits.shape[1]*3), dtype=np.uint8)

    for partner in lookup:  # A,B,C,D
        for id in partner:  # A1,A2,A3.....D4 etc
            recovery_bits[id] = np.concatenate(
                [average_bits[i] for i in partner if i != id])

    # shuffling here
    if key is not None and key != "":
        seed = np.frombuffer(
            key.encode('utf-8'), dtype=np.uint8)
        for i in range(recovery_bits.shape[0]):
            recovery_bits[i] = shuffle_under_seed(recovery_bits[i], seed)

    return recovery_bits
