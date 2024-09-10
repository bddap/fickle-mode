_:
    just -l

train:
    mkdir -p tmp
    uv run python -m fickle_mode.train --dataset-cache tmp/datasets
