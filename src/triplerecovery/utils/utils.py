import numpy as np
from math import log10, sqrt


def psnr(imarr_a: np.ndarray, imarr_b: np.ndarray):
    mse = np.mean((imarr_a - imarr_b) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


def set_bit(value: int, index: int, x: bool):
    # """Set the index:th bit of v to 1 if x is truthy, else to 0, and return the new value."""
    # mask = 1 << index   # Compute mask, an integer with just bit 'index' set.
    # # Clear the bit indicated by the mask (if x is False)
    # value &= ~mask
    # if x:
    #     # If x was True, set the bit indicated by the mask.
    #     value |= mask
    # return value            # Return the result, we're done.
    def set_bit2(value, bit):
        return value | (1 << bit)

    def clear_bit(value, bit):
        return value & ~(1 << bit)

    if x:
        return set_bit2(value, index)
    else:
        return clear_bit(value, index)


def get_bit(value: int, index: int):
    if value & (1 << index):
        return True
    else:
        return False


def set_lsb_zero(num: np.ndarray):
    '''
    Clearing the first two LSB of ndarray
    '''
    return set_bit(set_bit(num, 0, 0), 1, 0)


def shuffle_under_seed(ls: np.ndarray, seed: list[int]):
    lst = ls.copy()
    np.random.default_rng(seed).shuffle(lst)
    return lst


def unshuffle_list(shuffled_ls: np.ndarray, seed: list[int]):
    n = len(shuffled_ls)
    # Perm is [1, 2, ..., n]
    perm = list(range(1, n+1))
    # Apply sigma to perm
    shuffled_perm = shuffle_under_seed(perm, seed)
    # Zip and unshuffle
    zipped_ls = list(zip(shuffled_ls, shuffled_perm))
    zipped_ls.sort(key=lambda x: x[1])
    return np.array([a for (a, b) in zipped_ls])
