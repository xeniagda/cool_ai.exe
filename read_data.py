import math
import json
import os

import PIL.Image as im
import numpy as np
from scipy.misc import imresize

WANTED_SIZE = 32, 32

def read_data(path="compressed", amount=-1, verbose_level=1):
    images = []
    coords = []

    malformed_jsons = 0

    pngs = list(filter(lambda x: x.endswith(".png"), os.listdir(path)))

    if verbose_level == 1:
        print()
        amount_to_print = amount if amount != -1 else len(pngs)
        format_str = "[{:" + str(math.ceil(math.log10(amount_to_print))) + "d}/" + str(amount_to_print) + "]"
        format_str = "\033[1A" + format_str
    i = 0



    for f in sorted(pngs):
        if i == amount:
            break

        n = int(f[:-4])
        image = im.open(os.path.join(path, f))
        arr = (np.array(image.getdata(), dtype="float")/256).reshape(*image.size, -1).transpose(2, 0, 1)

        if arr.shape[1:] != WANTED_SIZE:
            try:
                arr = imresize(arr, WANTED_SIZE)
                continue
            except:
                if verbose_level > 1:
                    print("{} is of wrong dimensions: {}".format(f, arr.shape))

        arr = arr[:3,:,:]

        if arr.shape != (3, *WANTED_SIZE):
            if verbose_level > 1:
                print("{} is of wrong dimensions: expected {}, got {}".format(f, (3, *WANTED_SIZE), arr.shape))
            continue

        with open(os.path.join(path, str(n))) as coord_file:
            try:
                coord = json.loads(coord_file.read())
                lat, lng = coord["lat"], coord["lng"]
            except json.decoder.JSONDecodeError as e:
                if verbose_level > 1:
                    print("Malformed JSON in file {}: {}".format(n, e))
                malformed_jsons += 1
                continue

        images.append(arr)
        coords.append([lat, lng])
        i += 1

        if verbose_level == 1 and i % 13 == 0 or i == amount or i == len(pngs):
            print(format_str.format(i))

    if verbose_level == 1:
        print(format_str.format(amount_to_print), end="")
        print(", {} JSON files malformed".format(malformed_jsons))

    return np.array(images), np.array(coords)

def decode_lat_long(pos):
    return (pos[0] * 90, pos[1] * 180)

# From: https://www.reddit.com/r/geoguessr/comments/1ewrtu/points_vs_distance_data_trying_to_reverse/ca4ipl7/
def score(x):
    return 3400 * math.exp(-x**2/400000) + 1600 * math.exp(-x**2/40000000)

if __name__ == "__main__":
    import random
    import matplotlib.pyplot as plt
    from geopy.geocoders import Nominatim

    locator = Nominatim()

    images, coords = read_data(amount=500)
    print(images.shape)
    print(coords.shape)

    for i in range(10):
        idx = random.randrange(0, len(images))
        im = images[idx]
        co = coords[idx]
        lat, lng = decode_lat_long(co)
        loc = locator.reverse((lat, lng))
        print(co, loc)
        print(im.shape)
        plt.imshow(im.transpose(1, 2, 0))
        plt.show()