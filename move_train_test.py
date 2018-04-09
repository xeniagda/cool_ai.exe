import random
import shutil
import os

TRAIN_TEST_SPLIT = 0.1

TRAIN_IDX = 0
TEST_IDX = 0

for f in os.listdir(os.path.join("data", "tmp")):
    if not f.endswith(".png"):
        continue

    f = f[:-4]

    if not os.path.isfile(os.path.join("data", "tmp", f)):
        print(f, "failed")
        continue

    train = random.random() > TRAIN_TEST_SPLIT

    out = "train" if train else "test"
    out = os.path.join("data", out)

    while os.path.isfile(os.path.join(out, str(TRAIN_IDX if train else TEST_IDX))):
        if train:
            TRAIN_IDX += 1
        else:
            TEST_IDX += 1

    os.rename(os.path.join("data", "tmp", f), os.path.join(out, str(TRAIN_IDX if train else TEST_IDX)))
    os.rename(os.path.join("data", "tmp", f + ".png"), os.path.join(out, str(TRAIN_IDX if train else TEST_IDX) + ".png"))

