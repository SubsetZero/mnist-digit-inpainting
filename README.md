# MNIST Digit Inpainting

A small convolutional encoder-decoder that fills in a missing patch of a
handwritten digit image — given an MNIST digit with a chunk of it blanked out,
predict the full digit back.

<p align="center"><i>masked input &rarr; prediction &rarr; ground truth</i></p>

Idea sparked by an image-completion exercise I did in a university ML course,
but this is an independent implementation: built from scratch here, trained on
the public [MNIST](http://yann.lecun.com/exdb/mnist/) dataset (downloaded
automatically via `torchvision`), with no course code or data involved.

## The twist

Rather than always masking the same fixed, centered region, the mask here is a
**randomly placed square of randomly jittered size**, different for every
sample. The mask is also passed to the model as a second input channel, so it
has to use the mask to figure out *where* the gap is, not just learn a single
fixed hole shape.

## Model

A small U-Net-style encoder-decoder (`src/model.py`): two downsampling
conv blocks, a bottleneck, and two upsampling blocks with skip connections,
trained with a pixel-wise MSE loss that's weighted more heavily inside the
masked region.

## Project structure

```
src/
  dataset.py    # MaskedMNIST: wraps torchvision's MNIST with random masking
  model.py      # InpaintingCNN: the U-Net-style encoder-decoder
  train.py      # training loop (train/val/test split, early stopping)
  visualize.py  # save a grid of masked/predicted/ground-truth examples
```

## Usage

```sh
pip install -r requirements.txt

python src/train.py --epochs 20 --mask-size 10
python src/visualize.py --checkpoint ./checkpoints/best.pt --n 8
```

MNIST is downloaded automatically on first run. No pretrained weights or
precomputed outputs are included in this repo — everything here is meant to be
run, not just looked at.
