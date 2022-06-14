import cv2
import numpy as np
from PIL import Image
import time

from bits.recovery import make, embed, extract


def run(imagepath):
    start_t = time.time()

    imarr = np.array(Image.open(imagepath).convert('L'))
    imread_t = time.time()

    lookup = np.array([[0, 7, 13, 10],
                      [1, 6, 12, 11],
                      [4, 2, 9, 15],
                      [5, 3, 8, 14]], dtype=np.uint8)

    # make recoverybits
    recovery_bits = make(imarr, lookup)
    make_t = time.time()
    # embed recoverybits
    embededim = embed(imarr, recovery_bits)
    embed_t = time.time()
    # extract recoverybits
    extracted_bits = extract(embededim)
    extract_t = time.time()
    # compare

    # print
    print("imagepath:", imagepath)
    print("recovery_bits:", recovery_bits.shape)
    print("embededim:", embededim.shape)
    print("psnr", cv2.PSNR(imarr, embededim))
    print("extracted_bits:", extracted_bits.shape)
    print("imread_t:", imread_t-start_t)
    print("make_t:", make_t-imread_t)
    print("embed_t:", embed_t-make_t)
    print("extract_t:", extract_t-embed_t)
    print("result:", "CORRECT" if (recovery_bits !=
          extracted_bits).sum() == 0 else "INCORRECT")

    Image.fromarray(embededim).show("RecoveryBitTest")


imagepath = "/media/crackaf/44edb061-17db-4bf8-89a9-ad14bbb373b5/code/github/fyp/implementation/raw/lena.gif"
run(imagepath)
