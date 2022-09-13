# import random
import time
import zlib

from functools import partial
from collections import Counter
from operator import itemgetter

import numpy as np
import cv2

FILE_PARCHMENT = './parchment.png'
FILE_FONT = './papyrus.png'
FILE_MYSTERY = './mystery.png'

# the alphabet used
ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
# the 12 most common letters in English, approximately
COMMON12 = 'ETAOINSHRDLU'


def write(string, delay=.01):
    """Aesthetically pleasing delayed write function."""
    for s in string:
        print(s, end='', flush=True)
        time.sleep(delay)
    print()


def freq_distance(text):
    """A simple heuristic to compare a text string's letter frequencies with English letter
    frequency distribution. Computes the distance of the 12 most common letters in the given
    string from their location in the approximation of 12 most common letters in an average
    English corpus (ETAOIN SHRDLU)."""
    # get the frequencies of each letter in the text
    order = list(zip(*Counter(text).most_common()))[0]
    ret = 0

    for i, ch in enumerate(COMMON12):
        if ch not in order:
            loc = 13    # halfway down the alphabet is a fair default for unused letters
        else:
            loc = order.index(ch)
        ret += abs(i - loc)

    return ret


def extract_letters(img):
    """Extract out the letter shapes in the given image. The algorithm expects letters to be
    non-black on pure black background and separated from each other with at least one row or
    column of black pixels. First the algorithm splits the image horizontally into rows, then each
    row vertically into separate letter shapes."""
    # split rows
    rsums = img.sum(axis=1)
    ix = np.concatenate(([-1], np.where(rsums > 0)[0]))
    t1 = ix[np.where(np.diff(ix) != 1)[0] + 1] - 1
    jx = np.concatenate(([-1], np.where(rsums == 0)[0]))
    t2 = jx[np.where(np.diff(jx) != 1)[0] + 1] + 1

    blocks = []
    for y1, y2 in zip(t1, t2):
        blocks.append(img[y1:y2, :])

    letters = []
    # split each row into letters
    for block in blocks:
        csums = block.sum(axis=0)
        ix = np.concatenate(([-1], np.where(csums > 0)[0]))
        c1 = ix[np.where(np.diff(ix) != 1)[0] + 1] - 2
        jx = np.concatenate(([-1], np.where(csums == 0)[0]))
        c2 = jx[np.where(np.diff(jx) != 1)[0] + 1] + 2 

        for x1, x2 in zip(c1, c2):
            letter = block[:, x1:x2]
            letters.append(letter)

    return letters


def tight_crop(img):
    """Crop the image tightly by removing any black borders."""

    # compute row and column sums
    csums = img.sum(axis=0)
    rsums = img.sum(axis=1)

    # find rows and columns that are non-empty
    ci = np.where(csums > 0)[0]
    ri = np.where(rsums > 0)[0]

    # crop image to the first and last non-empty row and column
    # if the image is all zero, it all gets cropped!
    if ri.size == 0 or ci.size == 0:
        return img[:0, :0]

    img = img[ri[0]:ri[-1] + 1]
    img = img[:, ci[0]:ci[-1] + 1]
    return img


def read_font(fimg):
    """Read the font from an image and map each letter shape with its corresponding letter."""
    # extract letters and crop out any empty space around the letter shapes
    letters = map(tight_crop, extract_letters(fimg))

    return {ch: img.astype(np.uint8) for ch, img in zip(ALPHABET, letters)}


def resize(img, y):
    """Resize the given image so that its height is y and aspect ratio remains unchanged."""
    ratio = img.shape[1] / img.shape[0]
    x = int(ratio * y + .5)
    ret = cv2.resize(img, (x, y), interpolation=cv2.INTER_AREA)
    return ret 


def best_match(letter, font):
    """Compare this letter shape with each shape in the given font mapping using opencv's template
    matching and return the best match."""
    method = cv2.TM_CCOEFF_NORMED
    MAX = True  # change to False if using a minimum-best method (e.g. TM_SQDIFF)
    res = []

    # get the match value of best alignment for each letter in several font sizes
    for char, template in font.items():
        # compute several scalings of the font
        for y in range(letter.shape[0] - 5, letter.shape[0] + 1):
            rt = resize(template, y)
            if rt.shape[1] > letter.shape[1]: 
                continue
            m = cv2.matchTemplate(letter, rt, method)
            res.append((m.max() if MAX else m.min(), char, y))

    # return the absolutely best of the best matches
    res = sorted(res, reverse=MAX)
    return res[0]


def rot(msg, n):
    """Permute the given string msg by rotating each letter in it by n steps down the alphabet (e.g.
    n=3 for the classic Caesar cipher and -3 to reverse it)."""
    # the alphabet-rotation function parameterized by n
    _rot = lambda c, n: chr((ord(c) - 65 + n) % 26 + 65)
    func = partial(_rot, n=n)
    return ''.join(map(func, msg))
    

if __name__ == '__main__':
    write(' === Logging in...\n')
    time.sleep(1.5)

    # banner
    write('*' * 40)
    write('*    Hello, [anonymous]!               *')
    write('*    Welcome to W u n d e r b o x      *')
    write('*' * 40)

    write(' === Reading mysterious parchment...\n')

    # read the parchment containing the secret message and separate the R channel
    image = cv2.imread(FILE_PARCHMENT)[:, :, 0]
    # maximize contrast by scaling all values between 0 and 255
    image -= image.min()
    image *= 255 * (image.max() - image.min())

    write(' === Applying heat to make invisible ink visible...\n')
    # extract letter shapes from the image
    message = extract_letters(image)

    # read the font shapes from an image and generate a mapping from letters to them
    # reduce to two colors, maximize contrast and invert luminosity
    write(' === Analyzing handwriting...\n')
    font_image = 255 * (cv2.imread(FILE_FONT, 0) == 0)
    font = read_font(font_image)

    # ocr_chars = ['.'] * len(message)
    ocr_text = ''

    # find best match in the font alphabet for each extracted letter
    for i, letter in enumerate(message, 1):
        print('%d / %d...' % (i, len(message)), end='\r')
        letter = tight_crop(letter)
        _, char, _ = best_match(letter, font)
        ocr_text += char

    print()

    # generate all alphabet rotations of the message and compare their letter frequencies with that
    # of English
    # for i in range(26):
        # txt = rot(ocr_text, i)
        # print(freq_distance(txt), txt[:20])

    write(' === Deciphering encoded message...')

    distances = {}

    for i, ch in enumerate(ALPHABET, 1):
        perm = rot(ocr_text, i - 1)
        dist = freq_distance(perm)
        distances[perm] = dist
        print('[%s%s]' % (i * '.', (len(ALPHABET) - i) * ' '), end='\r')
        time.sleep(.02)

    print('\n')
    # invert the mapping to get the string with lowest value
    best, _ = sorted(distances.items(), key=itemgetter(1))[0]
    write(' Found the secret message:\n')
    write(best + '\n', .01)
    time.sleep(1.0)

    write(' === Investigating possible message sender...\n')
    for i in range(20):
        print('[' + ((i + 1) * '*') + ((19 - i) * ' ') + ']', end='\r')
        time.sleep(.1)
    print('\n')

    sender = zlib.decompress(cv2.imread(FILE_MYSTERY).ravel().tobytes()).decode('ascii')

    write(' *** SENDER IDENTIFIED ***')
    h = '-' * 24
    write('%s  TRANSMISSION STARTS   %s' % (h, h))

    write(sender, .0005)

    # for row in sender.split('\n'):
        # write(row, .001)
        # time.sleep(.05)

    write('%s  TRANSMISSION ENDS     %s\n' % (h, h))
    write(' === Process complete')
    print('Signed out automatically.')

