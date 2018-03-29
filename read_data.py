import math
import json
import os

import PIL.Image as im
from geopy.geocoders import Nominatim
import numpy as np
from scipy.misc import imresize

WANTED_SIZE = 32, 32

def read_data(path="compressed"):
    images = []
    coords = []

    for f in sorted(os.listdir(path)):
        if f.endswith(".png"):
            n = int(f[:-4])
            image = im.open(os.path.join(path, f))
            arr = (np.array(image.getdata(), dtype="float")/256).reshape(*image.size, -1).transpose(2, 0, 1)

            if arr.shape[1:] != WANTED_SIZE:
                arr = imresize(arr, WANTED_SIZE)
                continue

            arr = arr[:3,:,:]

            if arr.shape != (3, *WANTED_SIZE):
                print("{} is of wrong dimensions: expected {}, got {}".format(f, (3, *WANTED_SIZE), arr.shape))
                continue

            with open(os.path.join(path, str(n))) as coord_file:
                coord = json.loads(coord_file.read())
                lat, lng = coord["lat"], coord["lng"]

            images.append(arr)
            coords.append([lat, lng])

    return np.array(images), np.array(coords)

def decode_lat_long(pos):
    return (pos[0] * 90, pos[1] * 180)

# From: https://www.reddit.com/r/geoguessr/comments/1ewrtu/points_vs_distance_data_trying_to_reverse/ca4ipl7/
def score(x):
    return 3400 * math.exp(-x**2/400000) + 1600 * math.exp(-x**2/40000000)

if __name__ == "__main__":
    import random
    import matplotlib.pyplot as plt

    locator = Nominatim()

    images, coords = read_data()
    print(images.shape)
    print(coords.shape)

    for i in range(10):
        idx = random.randrange(0, len(images))
        im = images[idx]
        co = coords[idx]
        lat, lng = decode_lat_long(*co)
        loc = locator.reverse((lat, lng))
        print(co, loc)
        print(im.shape)
        plt.imshow(im.transpose(1, 2, 0))
        plt.show()