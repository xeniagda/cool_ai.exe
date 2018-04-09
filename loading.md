# Loading data

The data loader is a horrendous mess of `node`, `phantomjs` and `python3` that together loads data, resizes to 32x32 and splits into train and test.

## How use:

### Step 0: Create directories
Two temporary directories needs to be created in the `data` folder: `data/load` and `data/tmp`.

### Step 1: Load data using `fetchData.js`
This step takes screenshots from [GeoGuessr](https://www.geoguessr.com) and stores them in `data/load`, and coordinates, in `data/load`. This `phantomjs` script has a tendency to eat up all your RAM really quickly, so the script `fetchData.sh` is a 100% memory safe<sup>1</sup> wrapper.

The script will run and collect images, with a delay of ~30s to make sure everything gets loaded and rendered properly.

Dependencies: `phantomjs`.

### Step 2: Resize images using `resize.js`
The screenshots are all 1920x1080 which is way to big to be useful. Like before, this also leaks memory so the wrapper `resize.sh` will memory-secure this script.

Dependencies: `nodejs`, `node-image-resize` and `imagemagick`.

### Step 3: Split into train and test using `move_train_test.py`
Finally, the images have to be categorised into train and test. This script will do that, with 10% of images being test and the other 90% being train. It will not leak memory and therefore has no wrapper.

Dependencies: `python3`.

### TL;DR

```bash
mkdir data/load data/tmp
phantomjs fetchData.js
# ... <CTRL-C> when you're feeling done
node resize.js
python3 move_train_test.py
```

--
<sup>1: Not formally proved</sup>