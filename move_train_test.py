import random
import shutil
import os

TRAIN_TEST_SPLIT = 0.1

for f in os.listdir(os.path.join("data", "tmp")):
    if not f.endswith(".png"):
        continue

    f = f[:-4]

    if not os.path.isfile(os.path.join("data", "tmp", f)):
        print(f, "failed")
        continue


    out = "train" if random.random() > TRAIN_TEST_SPLIT else "test"
    out = os.path.join("data", out)

    os.rename(os.path.join("data", "tmp", f), os.path.join(out, f))
    os.rename(os.path.join("data", "tmp", f + ".png"), os.path.join(out, f + ".png"))

