import os
import numpy as np
import cv2
import time

from triplerecovery import bits, utils, embed, authenticate, recover

key = "this is key"
ekey = "this is key"


def test_grey_lena():
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
    recovery_bits = bits.recovery .make(imarr, lookup, key)
    lla = bits.recovery.embed(imarr, recovery_bits)
    hashes = bits.authentication.make(lla)

    # embedding
    ER = embed(imarr, 4, key=key)
    embedded = ER.imarr.copy()

    # extract recoverybits
    extracted_bits = bits.recovery.extract(embedded, ekey)
    # extract hashes
    exthashes = bits.authentication.extract(embedded)

    startx = 200
    starty = 200
    width = 100
    height = 100

    for i in range(width):
        for j in range(height):
            embedded[startx+i][starty+j] = 0
    # authenticate
    AU = authenticate(embedded)

    RE = recover(embedded, 4, key=ekey)

    print("Time: ", time.time() - start_t)
    print("RE sum: ", (recovery_bits != extracted_bits).sum())
    print("RE hash sum: ", (hashes != exthashes).sum())
    print("RE hash emb: ", (RE.imarr != embedded).sum())
    print("PSNR: ", utils.psnr(embedded, imarr))
    print("PSNR recoverd: ", utils.psnr(RE.imarr, imarr))

    # assert (recovery_bits != extracted_bits).sum(
    # ) == 0 and (hashes != exthashes).sum() == 0 and (
    #     RE.imarr != embedded).sum() == 12379 and (
    #         utils.psnr(embedded, imarr) == 39.97625935288754) and (
    #         utils.psnr(RE.imarr, imarr) == 41.322055472231476)

    return [imarr, ER.imarr, embedded, RE.imarr, AU.maskarr]


def test_lena():
    imagepath = os.path.dirname(os.path.abspath(__file__))+'/lena_color.gif'

    start_t = time.time()

    imarr = np.array(cv2.VideoCapture(imagepath).read()[1])
    imread_t = time.time()

    lookup = np.array([[0, 7, 13, 10],
                       [1, 6, 12, 11],
                       [4, 2, 9, 15],
                       [5, 3, 8, 14]], dtype=np.uint8)

    # make recoverybits
    recovery_bits = bits.recovery .make(imarr, lookup, key)
    lla = bits.recovery.embed(imarr, recovery_bits)
    hashes = bits.authentication.make(lla)

    # embedding
    ER = embed(imarr, 0, key=key)
    embedded = ER.imarr.copy()
    # authenticate
    # AU = authenticate(embedded)

    # extract recoverybits
    extracted_bits = bits.recovery.extract(embedded, ekey)
    # extract hashes
    exthashes = bits.authentication.extract(embedded)

    startx = 200
    starty = 200
    width = 100
    height = 100

    for i in range(width):
        for j in range(height):
            embedded[startx+i][starty+j] = 0

       # authenticate
    AU = authenticate(embedded)

    RE = recover(embedded, 0, key=ekey)

    print("Time: ", time.time() - start_t)
    print("RE sum: ", (recovery_bits != extracted_bits).sum())
    print("RE hash sum: ", (hashes != exthashes).sum())
    print("RE hash emb: ", (RE.imarr != embedded).sum())
    print("PSNR: ", utils.psnr(embedded, imarr))
    print("PSNR recoverd: ", utils.psnr(RE.imarr, imarr))

    # assert (recovery_bits != extracted_bits).sum(
    # ) == 0 and (hashes != exthashes).sum() == 0 and (
    #     RE.imarr != embedded).sum() == 37178 and (
    #         utils.psnr(embedded, imarr) == 40.495125793919584) and (
    #         utils.psnr(RE.imarr, imarr) == 41.57243355070828)

    return [imarr, ER.imarr, embedded, RE.imarr, AU.maskarr]


def test_grey_cat():
    imagepath = os.path.dirname(os.path.abspath(__file__))+'/cat.jpg'

    start_t = time.time()

    imarr = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)
    imread_t = time.time()

    lookup = np.array([[0, 7, 13, 10],
                       [1, 6, 12, 11],
                       [4, 2, 9, 15],
                       [5, 3, 8, 14]], dtype=np.uint8)

    # make recoverybits
    recovery_bits = bits.recovery .make(imarr, lookup, key)
    lla = bits.recovery.embed(imarr, recovery_bits)
    hashes = bits.authentication.make(lla)

    # embedding
    ER = embed(imarr, 0, key=key)
    embedded = ER.imarr.copy()
    # authenticate
    # AU = authenticate(embedded)

    # extract recoverybits
    extracted_bits = bits.recovery.extract(embedded, ekey)
    # extract hashes
    exthashes = bits.authentication.extract(embedded)

    startx = 200
    starty = 200
    width = 100
    height = 100

    for i in range(width):
        for j in range(height):
            embedded[startx+i][starty+j] = 0

    startx = 450
    starty = 300
    width = 200
    height = 100

    for i in range(width):
        for j in range(height):
            embedded[startx+i][starty+j] = 0

    startx = 400
    starty = 400
    width = 200
    height = 200

    for i in range(width):
        for j in range(height):
            embedded[startx+i][starty+j] = 0

       # authenticate
    AU = authenticate(embedded)

    RE = recover(embedded, 0, key=ekey)

    print("Time: ", time.time() - start_t)
    print("RE sum: ", (recovery_bits != extracted_bits).sum())
    print("RE hash sum: ", (hashes != exthashes).sum())
    print("RE hash emb: ", (RE.imarr != embedded).sum())
    print("PSNR: ", utils.psnr(embedded, imarr))
    print("PSNR recoverd: ", utils.psnr(RE.imarr, imarr))

    # assert (recovery_bits != extracted_bits).sum(
    # ) == 0 and (hashes != exthashes).sum() == 0 and (
    #     RE.imarr != embedded).sum() == 24965 and (
    #         utils.psnr(embedded, imarr) == 42.25235578920841) and (
    #         utils.psnr(RE.imarr, imarr) == 42.5450185633773)

    return [imarr, ER.imarr, embedded, RE.imarr, AU.maskarr]


def test_cat():
    imagepath = os.path.dirname(os.path.abspath(__file__))+'/cat.jpg'

    start_t = time.time()

    imarr = cv2.imread(imagepath, cv2.IMREAD_COLOR)
    imread_t = time.time()

    lookup = np.array([[0, 7, 13, 10],
                       [1, 6, 12, 11],
                       [4, 2, 9, 15],
                       [5, 3, 8, 14]], dtype=np.uint8)

    # make recoverybits
    recovery_bits = bits.recovery .make(imarr, lookup, key)
    lla = bits.recovery.embed(imarr, recovery_bits)
    hashes = bits.authentication.make(lla)

    # embedding
    ER = embed(imarr, 0, key=key)
    embedded = ER.imarr.copy()
    # authenticate
    # AU = authenticate(embedded)

    # extract recoverybits
    extracted_bits = bits.recovery.extract(embedded, ekey)
    # extract hashes
    exthashes = bits.authentication.extract(embedded)

    startx = 200
    starty = 200
    width = 100
    height = 100

    for i in range(width):
        for j in range(height):
            embedded[startx+i][starty+j] = 0

    startx = 400
    starty = 400
    width = 200
    height = 200

    for i in range(width):
        for j in range(height):
            embedded[startx+i][starty+j] = 0

       # authenticate
    AU = authenticate(embedded)

    RE = recover(embedded, 1, key=ekey)

    print("Time: ", time.time() - start_t)
    print("RE sum: ", (recovery_bits != extracted_bits).sum())
    print("RE hash sum: ", (hashes != exthashes).sum())
    print("RE hash emb: ", (RE.imarr != embedded).sum())
    print("PSNR: ", utils.psnr(embedded, imarr))
    print("PSNR recoverd: ", utils.psnr(RE.imarr, imarr))

    # assert (recovery_bits != extracted_bits).sum(
    # ) == 0 and (hashes != exthashes).sum() == 0 and (
    #     RE.imarr != embedded).sum() == 37202 and (
    #         utils.psnr(embedded, imarr) == 43.13475519968526) and (
    #         utils.psnr(RE.imarr, imarr) == 43.46553215937854)

    return [imarr, ER.imarr, embedded, RE.imarr, AU.maskarr]


if True:
    [imarr, imarr2, imarr3, imarr4, imarr5] = test_grey_lena()
    cv2.imshow("Original", imarr)
    cv2.imshow("Embedded", imarr2)
    cv2.imshow("Edited", imarr3)
    cv2.imshow("Recovered", imarr4)
    cv2.imshow("MASK", imarr5)

    cv2.waitKey(0)
