import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import read_data

ERASE_LAST = "\033[1A\033[K"

"""
IN -> -1x3x32x32
conv 10 3x3 -> -1x10x30x30
pool 3 -> -1x10x10x10
conv 3 3x3 -> -1x5x8x8
pool 4 -> -1x5x2x2
connected 2 -> -1x2
"""

SAVE_PATH = "ai.tar"
BEST_PATH = "ai_best.tar"

TRAIN_TEST_SPLIT = 0.8 # 80% train, 20% test
LEARNING_RATE = 0.2
MOMENTUM = 0.3

class cool_ai_exe(nn.Module):
    def __init__(self):
        super(cool_ai_exe, self).__init__()

        self.conv_1 = nn.Conv2d(3, 10, (3, 3))
        self.conv_2 = nn.Conv2d(10, 5, (3, 3))
        self.conn   = nn.Linear(20, 2)

    def forward(self, x):
        x = self.conv_1(x)
        x = nn.MaxPool2d((3, 3))(x)
        x = self.conv_2(x)
        x = nn.MaxPool2d((4, 4))(x)
        x = x.view(x.shape[0], -1)
        x = self.conn(x)

        return x


if __name__ == "__main__":
    print("Loading data...")
    data, coords = read_data.read_data(amount=5000)
    amount = int(data.shape[0] * TRAIN_TEST_SPLIT)
    data_train, data_test = data[:amount], data[amount:]
    coords_train, coords_test = coords[:amount], coords[amount:]

    print("Train images:", data_train.shape)
    print("Test images:", data_test.shape)

    ai = cool_ai_exe()

    crit = nn.MSELoss()
    opt = optim.SGD(ai.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    EPOCH = 0

    train_loss_history = []
    test_loss_history = []

    if os.path.isfile(SAVE_PATH):
        print("Loading old state")
        state = torch.load(SAVE_PATH)
        ai = state["ai"]
        opt = state["opt"]
        EPOCH = state["epoch"]
        train_loss_history = state["train_loss_history"]
        test_loss_history  = state["test_loss_history"]

    while True:
        EPOCH += 1

        print("Generation {:4d}".format(EPOCH))
        ai.zero_grad()

        # print("Training...")
        res = ai(Variable(torch.Tensor(data_train)))
        # print(ERASE_LAST + "Generated...")

        loss = crit(res, Variable(torch.Tensor(coords_train)))
        # print("Loss:", loss.data[0])
        loss.backward()
        # print(ERASE_LAST + "Backwarded")

        opt.step()
        # print(ERASE_LAST + "Stepped")

        test_res = ai(Variable(torch.Tensor(data_test)))
        test_loss = crit(test_res, Variable(torch.Tensor(coords_test)))
        print(ERASE_LAST +
                "Generation {:4d}: Test: {:.4f}, Train: {:.4f}".format(EPOCH, test_loss.data[0], loss.data[0]))

        train_loss_history.append(loss.data[0])
        test_loss_history.append(test_loss.data[0])

        try:
            torch.save(
                    {"ai": ai, "opt": opt, "epoch": EPOCH, "train_loss_history": train_loss_history
                    , "test_loss_history": test_loss_history}, SAVE_PATH)

            if loss.data[0] <= min(train_loss_history):
                torch.save(
                        {"ai": ai, "opt": opt, "epoch": EPOCH, "train_loss_history": train_loss_history
                        , "test_loss_history": test_loss_history}, BEST_PATH)
                print("Best!")
        except KeyboardInterrupt as e:
            torch.save(
                    {"ai": ai, "opt": opt, "epoch": EPOCH, "train_loss_history": train_loss_history
                    , "test_loss_history": test_loss_history}, SAVE_PATH)

            if loss.data[0] <= min(train_loss_history):
                torch.save(
                        {"ai": ai, "opt": opt, "epoch": EPOCH, "train_loss_history": train_loss_history
                        , "test_loss_history": test_loss_history}, BEST_PATH)
                print("Best!")
            raise e
