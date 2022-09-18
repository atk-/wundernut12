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


def write(string, delay=.005):
    """Aesthetically pleasing delayed write function.
    :param string: string to write
    :param delay: the delay between consecutive letters
    """
    for s in string:
        print(s, end='', flush=True)
        time.sleep(delay)
        # wait for longer on line breaks
        if s == '\n':
            time.sleep(5 * delay)
    print()


def freq_distance(text):
    """A simple heuristic to compare a text string's letter frequencies with English letter
    frequency distribution. Computes the distance of the 12 most common letters in the given
    string from their location in the approximation of 12 most common letters in an average
    English corpus (ETAOIN SHRDLU).

    :param text: the text string to analyze
    :returns: the computed distance of the given string's letter distribution to English on
    average
    """
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
    row vertically into separate letter shapes.

    :param img: image to process
    :returns: a list of letter shapes detected from the image
    """
    # first split image into separate rows
    rsums = img.sum(axis=1)
    ix = np.concatenate(([-1], np.where(rsums > 0)[0]))
    t1 = ix[np.where(np.diff(ix) != 1)[0] + 1] - 1
    jx = np.concatenate(([-1], np.where(rsums == 0)[0]))
    t2 = jx[np.where(np.diff(jx) != 1)[0] + 1] + 1

    blocks = []
    for y1, y2 in zip(t1, t2):
        blocks.append(img[y1:y2, :])

    letters = []
    # then split each row into letters
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
    """Crop the image tightly by removing any black borders.

    :param img: the image to process
    :returns: the cropped image
    """

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

    img = img[ri[0]:ri[-1] + 1, ci[0]:ci[-1] + 1]
    return img


def read_font(fname):
    """Read the font from an image and map each letter shape with its corresponding letter.

    :param fname: the image file to read
    :returns: a mapping from letters to their images
    """
    # read image, reduce to two colors, maximize contrast and invert luminosity
    font_image = 255 * (cv2.imread(fname, 0) == 0)
    # extract letters and crop out any empty space around the letter shapes
    letters = map(tight_crop, extract_letters(font_image))

    return {ch: img.astype(np.uint8) for ch, img in zip(ALPHABET, letters)}


def resize(img, y):
    """Resize the given image so that its height is y and aspect ratio remains unchanged.

    :param img: the image to resize
    :param y: the height to scale to
    :returns: the resized image
    """
    ratio = img.shape[1] / img.shape[0]
    x = int(ratio * y + .5)
    ret = cv2.resize(img, (x, y), interpolation=cv2.INTER_AREA)
    return ret


def best_match(letter, font):
    """Compare this letter shape with each shape in the given font mapping using opencv's template
    matching engine and return the best match.
    :param letter: the letter shape
    :param font: the font mapping to compare with
    :returns: a tuple of (score, letter, height) of the best letter match
    """
    method = cv2.TM_CCOEFF_NORMED
    MAX = True  # change to False if using a minimum-best method (e.g. TM_SQDIFF)
    res = []

    # get the match value of best alignment for each letter in several font sizes
    for char, template in font.items():
        # try several scalings of the font
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
    """Permute the given string msg by rotating each letter in it by n steps along the alphabet (e.g.
    n=3 for the classic Caesar cipher and -3 to reverse it).

    :param msg: the string to permute
    :param n: the number of steps to rotate each letter along the alphabet
    """

    def rotate(c, n):
        """Rotate the character c down n steps along the capital letters A-Z."""
        return chr((ord(c) - ord(ALPHABET[0]) + n) % len(ALPHABET) + ord(ALPHABET[0]))

    func = partial(rotate, n=n)
    return ''.join(map(func, msg))


def print_banner(*lines):
    """Print out a banner text.

    :param lines: lines to write inside a banner
    """
    width = max(map(len, lines))
    write('*' * (width + 10))
    for line in lines:
        write('*    %s    *' % line.ljust(width))
    write('*' * (width + 10))


def read_parchment(fname):
    """Read the parchment image.

    :param fname: the filename to read
    :returns: the parchment image
    """
    # read the parchment containing the secret message and separate the R channel
    image = cv2.imread(fname)[:, :, 0]
    # maximize contrast by scaling all values between 0 and 255
    image -= image.min()
    image *= 255 * (image.max() - image.min())
    return image


if __name__ == '__main__':
    write(' === Logging in...\n')
    time.sleep(1.5)

    print_banner('Hello, anonymous!', 'Welcome to W U N D E R B O X')

    write(' === Reading the mysterious parchment...\n')
    parchment = read_parchment(FILE_PARCHMENT)

    write(' === Applying heat to make invisible ink visible...\n')
    # extract letter shapes from the image
    message = extract_letters(parchment)
    write('%d letter symbols appear on the parchment!' % len(message))

    write(' === Installing the Papyrus typeface...')
    # read the font shapes from an image and generate a mapping from letters to them
    font = read_font(FILE_FONT)

    write(' === Analyzing handwriting...\n')
    ocr_text = ''

    # find best match in the font alphabet for each extracted letter
    for i, letter in enumerate(message, 1):
        print('%d / %d letters decoded...' % (i, len(message)), end='\r')
        letter = tight_crop(letter)
        _, char, _ = best_match(letter, font)
        ocr_text += char

    print()

    write(' === Deciphering encoded message...')

    distances = {}

    # go through alphabetical rotations of the string and evaluate them
    for i, ch in enumerate(ALPHABET, 1):
        perm = rot(ocr_text, i - 1)
        dist = freq_distance(perm)
        distances[perm] = dist
        print('[%s%s]' % (i * '.', (len(ALPHABET) - i) * ' '), end='\r')

    print('\n')

    # sort dictionary by values to get the item with the lowest (and hence the best) score
    best, _ = sorted(distances.items(), key=itemgetter(1))[0]
    write(' Found the secret message:\n')
    write(best + '\n', .01)

    write(' === Investigating possible message sender...\n')
    for i in range(20):
        print('[' + ((i + 1) * '*') + ((19 - i) * ' ') + ']', end='\r')
        time.sleep(.05)
    print('\n')

    sender = zlib.decompress(cv2.imread(FILE_MYSTERY).ravel().tobytes()).decode('ascii')

    write(' *** SENDER IDENTIFIED ***')
    h = '-' * 24
    write('%s  TRANSMISSION STARTS   %s' % (h, h))
    write(sender, .0002)
    write('%s  TRANSMISSION ENDS     %s\n' % (h, h))
    write(' === Process complete')

    print('Signed out automatically.')
