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

 - Python 3.9+
 - The Python packages listed in `requirements.txt` (NumPy, OpenCV and `numba`)

## Installation

Install the dependencies (including `numba`) with:

```
pip install -r requirements.txt
```

In a Codex workspace this setup is automated by running `codex/setup.sh`,
which installs the required Python packages.

Alternatively the project can be installed as a package using ``pip``:

```
pip install .
```

Installing the package exposes a ``kfe-codec`` command that wraps
``python kfe_codec.py`` for convenient use from the command line.

## Usage

Encode a binary file to a video. By default the output is an MKV container
encoded with FFV1. Use ``--container mp4`` to produce an MP4 file:

```
kfe-codec encode bin/input.bin kfe/output --container mkv
```

Decode a video back to a binary file:

```
kfe-codec decode kfe/output.mkv bin/restored.bin
```

Add ``--progress`` to either command to display progress information during
encoding or decoding:

```
kfe-codec encode bin/input.bin kfe/output --progress
```

The ``kfe-codec`` command becomes available after installing the package with
``pip install .``. Direct execution with ``python kfe_codec.py`` continues to
work if preferred.

The codec uses frames of size 3840×2160 (RGB), so each frame stores exactly
24,883,200 bytes of data. The original file size is written to the first frame
along with a SHA-256 checksum so any padding added to the final frame can be
removed during decoding and the output validated.

The implementation uses the **FFV1** codec for writing videos. FFV1 is a
lossless codec, ensuring that every byte of the original file is preserved in
the encoded video without degradation.

## Certificate-based Encryption

cpECSK performs a deterministic permutation of pixels in every frame. The
permutation parameters are derived from a user supplied *certificate* file. Any
non-empty file can act as a certificate – for example one created with
``dd if=/dev/urandom of=my.cert bs=32 count=1``. Pass this file with the
``--cert`` option when encoding:

```
kfe-codec encode bin/input.bin kfe/output --cert my.cert
```

The SHA‑256 digest of the certificate is written to the header frame and a copy
of the certificate is saved alongside the video as ``<output>.cert``. During
decoding the same certificate must be supplied. If ``<video>.cert`` exists in
the same directory it is loaded automatically, otherwise provide the path
explicitly:

```
kfe-codec decode kfe/output.mkv bin/restored.bin --cert my.cert
```

Decoding fails if the certificate's digest does not match the embedded value.

## Testing

Once the dependencies are installed you can run the automated tests with:

```
pytest
```

The suite exercises encoding and decoding behavior, including the MP4 fallback
mechanism.
