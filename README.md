# KFE Codec Prototype

This repository contains a basic prototype implementation of the KFE codec as
outlined in the specification. The codec converts binary files into 4K video
files and can reverse the process to reconstruct the original binary data.

## Directory Structure

```
bin/    # input and output binary files
kfe/    # generated video files
kfe_codec.py  # codec implementation
```

## Requirements

- Python 3.8+
- NumPy
- OpenCV (`opencv-python`)

Install the Python dependencies with:

```
pip install numpy opencv-python
```

## Usage

Encode a binary file to a video:

```
python kfe_codec.py encode bin/input.bin kfe/output.mp4
```

Decode a video back to a binary file:

```
python kfe_codec.py decode kfe/output.mp4 bin/restored.bin
```

The codec uses frames of size 3840×2160 (RGB), so each frame stores exactly
24,883,200 bytes of data. The original file size is written to the first frame
so any padding added to the final frame can be removed during decoding.
