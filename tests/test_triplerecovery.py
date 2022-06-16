import os
import numpy as np
import cv2
import time

from triplerecovery import bits, utils, embed, authenticate, recover


def test_grey():
    imagepath = os.path.dirname(os.path.abspath(__file__))+'/lena.gif'

    start_t = time.time()

    imarr = np.array(cv2.cvtColor(
        cv2.VideoCapture(imagepath).read()[1], cv2.COLOR_BGR2GRAY))
    imread_t = time.time()

    lookup = np.array([[0, 7, 13, 10],
                      [1, 6, 12, 11],
                      [4, 2, 9, 15],
                      [5, 3, 8, 14]], dtype=np.uint8)

    # make recoverybits
    recovery_bits = bits.recovery .make(imarr, lookup)
    lla = bits.recovery.embed(imarr, recovery_bits)
    hashes = bits.authentication.make(lla)

    # embedding
    ER = embed(imarr, lookup)
    embedded = ER.imarr.copy()
    # authenticate
    AU = authenticate(embedded)

    # extract recoverybits
    extracted_bits = bits.recovery.extract(embedded)
    # extract hashes
    exthashes = bits.authentication.extract(embedded)

    startx = 200
    starty = 200
    width = 100
    height = 100

    for i in range(width):
        for j in range(height):
            embedded[startx+i][starty+j] = 0

    RE = recover(embedded, lookup)

    print("Time: ", time.time() - start_t)
    # cv2.imshow("Original", imarr)
    # cv2.imshow("Embedded",ER.imarr)
    # cv2.imshow("Edited",embedded)
    # cv2.imshow("Recovered",RE.imarr)

    # cv2.waitKey(0)

    assert (recovery_bits != extracted_bits).sum(
    ) == 0 and (hashes != exthashes).sum() == 0 and (
        RE.imarr != embedded).sum() == 12379 and (
            utils.psnr(embedded, imarr) == 39.97625935288754) and (
            utils.psnr(RE.imarr, imarr) == 41.322055472231476)

    return [imarr, ER.imarr, embedded, RE.imarr]


def test_rgb():
    imagepath = os.path.dirname(os.path.abspath(__file__))+'/lena_color.gif'

    start_t = time.time()

    imarr = np.array(cv2.VideoCapture(imagepath).read()[1])
    imread_t = time.time()

    lookup = np.array([[0, 7, 13, 10],
                      [1, 6, 12, 11],
                      [4, 2, 9, 15],
                      [5, 3, 8, 14]], dtype=np.uint8)

    # make recoverybits
    recovery_bits = bits.recovery .make(imarr, lookup)
    lla = bits.recovery.embed(imarr, recovery_bits)
    hashes = bits.authentication.make(lla)

    # embedding
    ER = embed(imarr, lookup)
    embedded = ER.imarr.copy()
    # authenticate
    # AU = authenticate(embedded)

    # extract recoverybits
    extracted_bits = bits.recovery.extract(embedded)
    # extract hashes
    exthashes = bits.authentication.extract(embedded)

    startx = 200
    starty = 200
    width = 100
    height = 100

    for i in range(width):
        for j in range(height):
            embedded[startx+i][starty+j] = 0

    RE = recover(embedded, lookup)

    print("Time: ", time.time() - start_t)
    # cv2.imshow("Original", imarr)
    # cv2.imshow("Embedded", ER.imarr)
    # cv2.imshow("Edited", embedded)
    # cv2.imshow("Recovered", RE.imarr)

    # cv2.waitKey(0)

    assert (recovery_bits != extracted_bits).sum(
    ) == 0 and (hashes != exthashes).sum() == 0 and (
        RE.imarr != embedded).sum() == 37178 and (
            utils.psnr(embedded, imarr) == 40.495125793919584) and (
            utils.psnr(RE.imarr, imarr) == 41.57243355070828)
    return [imarr, ER.imarr, embedded, RE.imarr]


# values = test_rgb()
# cv2.imshow("Original", imarr)
# cv2.imshow("Embedded", ER.imarr)
# cv2.imshow("Edited", embedded)
# cv2.imshow("Recovered", RE.imarr)

# cv2.waitKey(0)
