import numpy as np
import cv2
from triplerecovery.blocks import make as make_blocks
from triplerecovery.utils import shuffle_under_seed

'''
Will make the recovery bits for image
'''


def make(imarr: np.ndarray, lookup: np.ndarray, key: str) -> np.ndarray:
    '''
    '''

    zoomx = 4  # according to the paper the zoom factor is 4
    averages = cv2.resize(imarr, None, fx=1/zoomx, fy=1 /
                          zoomx, interpolation=cv2.INTER_AREA)
    averages = make_blocks(
        averages, (averages.shape[0]//zoomx, averages.shape[1]//zoomx), addChannel=False).reshape(16, -1)

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
