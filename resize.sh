while [ ! -z "$(ls data)" ]; do
    node resize.js 5
done
