import math
import json
import os

import PIL.Image as im
from geopy.geocoders import Nominatim
import numpy as np
from scipy.misc import imresize

WANTED_SIZE = 32, 32

def read_data(path="compressed", amount=-1):
    images = []
    coords = []

    files = os.listdir(path)
    if amount != -1:
        files = files[:amount]

    for f in sorted(files):
        if f.endswith(".png"):
            n = int(f[:-4])
            image = im.open(os.path.join(path, f))
            arr = (np.array(image.getdata(), dtype="float")/256).reshape(*image.size, -1).transpose(2, 0, 1)

            if arr.shape[1:] != WANTED_SIZE:
                try:
                    arr = imresize(arr, WANTED_SIZE)
                    continue
                except:
                    print("{} is of wrong dimensions: {}".format(f, arr.shape))

            arr = arr[:3,:,:]

            if arr.shape != (3, *WANTED_SIZE):
                print("{} is of wrong dimensions: expected {}, got {}".format(f, (3, *WANTED_SIZE), arr.shape))
                continue

            with open(os.path.join(path, str(n))) as coord_file:
                try:
                    coord = json.loads(coord_file.read())
                    lat, lng = coord["lat"], coord["lng"]
                except json.decoder.JSONDecodeError as e:
                    print("Malformed JSON in file {}: {}".format(n, e))
                    continue

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

    images, coords = read_data(amount=5000)
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