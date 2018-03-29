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

data, coords = read_data.read_data()
amount = int(data.shape[0] * TRAIN_TEST_SPLIT)

data_test   = data[amount:] if USE_TEST else data[:amount]
coords_test = coords[amount:] if USE_TEST else coords[:amount]

print("Test images:", data_test.shape)

ai = cool_ai_exe()

EPOCH = 0

if os.path.isfile(BEST_PATH):
    print("Loading old state")
    state = torch.load(BEST_PATH)
    ai = state["ai"]
    opt = state["opt"]
    EPOCH = state["epoch"]
    train_loss_history = state["train_loss_history"]
    test_loss_history  = state["test_loss_history"]
else:
    print("No ai save")
    exit()

print("Epoch", EPOCH)

train, = plt.plot(train_loss_history, label="Train loss")
test, = plt.plot(test_loss_history, label="Test loss")
plt.legend(handles=[train, test])
plt.show()


locator = Nominatim()

for i in range(10):
    idx = random.randrange(len(data_test))

    im = data_test[idx]
    coord = coords_test[idx]

    guess = ai(torch.autograd.Variable(torch.Tensor([im]))).data.numpy()[0]

    coord_ll = list(read_data.decode_lat_long(coord))
    guess_ll = list(read_data.decode_lat_long(guess))

    dist = distance(coord_ll, guess_ll).kilometers

    print("Im #{}: guess: ({:.4f}, {:.4f}), real: ({:.4f}, {:.4f})".format(i, *guess_ll, *coord_ll))
    try: print("Guess address:", locator.reverse(guess_ll))
    except: pass
    try: print("Real address:", locator.reverse(coord_ll))
    except: pass
    print("Distance: {:.4f}km".format(dist))
    print("Score: {:d}".format(round(read_data.score(dist))))

    plt.imshow(im.transpose(1, 2, 0))
    plt.show()
