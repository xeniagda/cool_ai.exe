import os
import sys

import argparse
# Parse arguments
parser = argparse.ArgumentParser(description="Train the cool_ai.exe")
parser.add_argument(
        "--load-path", "-l",
        type=str, default="ai.tar",
        help="Where to load the cool_ai.exe from")
parser.add_argument(
        "--save-path", "-s",
        type=str, default="ai.tar",
        help="Where to save the cool_ai.exe to")
parser.add_argument(
        "--best-path", "-b",
        type=str, default="ai_best.tar",
        help="Where to save the best cool_ai.exe")
parser.add_argument(
        "--amount",
        type=int,
        help="How many images to train on. Leave blank for all")
parser.add_argument(
        "--split",
        type=float, default=0.8,
        help="The train/test split")
parser.add_argument(
        "--learning-rate", "-lr",
        type=float, default=0.2,
        help="The learning rate the cool_ai.exe should use")
parser.add_argument(
        "--momentum", "-m",
        type=float, default=0.3,
        help="Learning momentum")
parser.add_argument(
        "--new", "-n",
        action="store_true",
        help="Create a new cool_ai.exe")

args = parser.parse_args()

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
connected 20 -> -1x20
connected 10 -> -1x10
connected 2 -> -1x2
"""

SAVE_PATH = args.save_path
LOAD_PATH = args.load_path
BEST_PATH = args.best_path

TRAIN_TEST_SPLIT = args.split
LEARNING_RATE = args.learning_rate
MOMENTUM = args.momentum




class cool_ai_exe(nn.Module):
    def __init__(self):
        super(cool_ai_exe, self).__init__()

        self.conv_1 = nn.Conv2d(3, 10, (3, 3))
        self.conv_2 = nn.Conv2d(10, 5, (3, 3))
        self.conn_1 = nn.Linear(20, 20)
        self.conn_2 = nn.Linear(20, 10)
        self.conn_3 = nn.Linear(10, 2)

    def forward(self, x):
        x = self.conv_1(x)
        x = nn.MaxPool2d((3, 3))(x)
        x = self.conv_2(x)
        x = nn.MaxPool2d((4, 4))(x)

        x = x.view(x.shape[0], -1)
        x = self.conn_1(x)
        x = self.conn_2(x)
        x = self.conn_3(x)

        return x


if __name__ == "__main__":
    print("Loading data...")
    data, coords = read_data.read_data(amount=args.amount or -1)
    amount = int(data.shape[0] * TRAIN_TEST_SPLIT)
    data_train, data_test = data[:amount], data[amount:]
    coords_train, coords_test = coords[:amount], coords[amount:]

    print("Train images:", " x ".join(map(str, data_train.shape)))
    print(" Test images:", " x ".join(map(str, data_test.shape)))

    ai = cool_ai_exe()

    crit = nn.MSELoss()
    opt = optim.SGD(ai.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    EPOCH = 0

    train_loss_history = []
    test_loss_history = []

    if not args.new:
        if os.path.isfile(LOAD_PATH):
            print("Loading old state")
            state = torch.load(LOAD_PATH)
            ai = state["ai"]
            opt = state["opt"]
            EPOCH = state["epoch"]
            train_loss_history = state["train_loss_history"]
            test_loss_history  = state["test_loss_history"]
        else:
            print("Can't find {}. If you want to create a new ai, use --new".format(LOAD_PATH))

    while True:
        EPOCH += 1

        print("Generation {:4d}...".format(EPOCH))
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

        train_loss_history.append(loss.data[0])
        test_loss_history.append(test_loss.data[0])

        is_best = loss.data[0] == min(train_loss_history)
        if is_best:
            print(ERASE_LAST +
                "Generation {:4d}: Test: {:.4f}, Train: {:.4f} (best)".format(EPOCH, test_loss.data[0], loss.data[0]))
        else:
            print(ERASE_LAST +
                "Generation {:4d}: Test: {:.4f}, Train: {:.4f}".format(EPOCH, test_loss.data[0], loss.data[0]))

        try:
            torch.save(
                    {"ai": ai, "opt": opt, "epoch": EPOCH, "train_loss_history": train_loss_history
                    , "test_loss_history": test_loss_history}, SAVE_PATH)

            if is_best:
                torch.save(
                        {"ai": ai, "opt": opt, "epoch": EPOCH, "train_loss_history": train_loss_history
                        , "test_loss_history": test_loss_history}, BEST_PATH)
        except KeyboardInterrupt as e:
            torch.save(
                    {"ai": ai, "opt": opt, "epoch": EPOCH, "train_loss_history": train_loss_history
                    , "test_loss_history": test_loss_history}, SAVE_PATH)

            if is_best:
                torch.save(
                        {"ai": ai, "opt": opt, "epoch": EPOCH, "train_loss_history": train_loss_history
                        , "test_loss_history": test_loss_history}, BEST_PATH)
            raise e

