import os
import random

import matplotlib.pyplot as plt
import numpy as np
from geopy.geocoders import Nominatim
from geopy.distance import distance

from ai import *
import read_data
import torch

USE_TEST = False
TEST_BEST = False
proj = plt.imread("mercator.png")

EPOCH = 0

path_to_use = BEST_PATH if TEST_BEST else SAVE_PATH
if os.path.isfile(path_to_use):
    print("Loading state from", path_to_use)
    state = torch.load(path_to_use)
    ai = state["ai"]
    opt = state["opt"]
    EPOCH = state["epoch"]
    train_loss_history = state["train_loss_history"]
    test_loss_history  = state["test_loss_history"]
    # amount = state["amount"]
else:
    print("No ai save")
    exit()

if len(sys.argv) == 2:
    nr_to_show = int(sys.argv[1])
else:
    nr_to_show = 10

data_test, coords_test = read_data.read_data(not USE_TEST, amount=100)

print("Test images:", data_test.shape)

print("Epoch", EPOCH)

train, = plt.plot(train_loss_history, label="Train loss")
test, = plt.plot(test_loss_history, label="Test loss")
plt.legend(handles=[train, test])
plt.show()


locator = Nominatim()

def long_lat2xy(long, lat):
    x = (lat + 180) / 360 * proj.shape[1]
    y = (-long + 110) / 108 / 2 * proj.shape[0]
    return x, y

while True:
    for i in range(nr_to_show):
        idx = random.randrange(len(data_test))

        im = data_test[idx]
        coord = coords_test[idx]

        guess = ai(torch.autograd.Variable(torch.Tensor([im]))).data.numpy()[0]

        coord_ll = list(read_data.decode_lat_long(coord))
        guess_ll = list(read_data.decode_lat_long(guess))

        dist = distance(coord_ll, guess_ll).kilometers

        print("Im #{}: guess: ({:.4f}, {:.4f}), real: ({:.4f}, {:.4f})".format(i, *guess_ll, *coord_ll))
        try: print("Guess address: " + str(locator.reverse(guess_ll)))
        except: pass
        try: print("Real address: " +  str(locator.reverse(coord_ll)))
        except: pass
        print("Distance: {:.4f}km".format(dist))
        print("Score: {:d}".format(round(read_data.score(dist))))

        plt.subplot(121)
        plt.imshow(im.transpose(1, 2, 0))

        plt.subplot(122)
        plt.imshow(proj)
        plt.scatter(*long_lat2xy(*coord_ll), c="green")
        plt.scatter(*long_lat2xy(*guess_ll), c="red")

        plt.show()

    if input("Continue? [Y/n] ").lower() == "n":
        break
