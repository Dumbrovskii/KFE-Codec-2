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
- The Python packages listed in `requirements.txt`

Install the dependencies with:

```
pip install -r requirements.txt
```

In a Codex workspace this setup is automated by running `codex/setup.sh`,
which installs the required Python packages.

## Usage

Encode a binary file to a video (the video is encoded using the lossless FFV1
codec):

```
python kfe_codec.py encode bin/input.bin kfe/output.mkv
```

When the output path ends in `.mp4` the encoder first writes a temporary MKV
file and then renames it to the requested MP4 filename. This allows the lossless
FFV1 stream to be stored inside an MP4 container.

Decode a video back to a binary file:

```
python kfe_codec.py decode kfe/output.mkv bin/restored.bin
```

The codec uses frames of size 3840×2160 (RGB), so each frame stores exactly
24,883,200 bytes of data. The original file size is written to the first frame
so any padding added to the final frame can be removed during decoding.

The implementation uses the **FFV1** codec for writing videos. FFV1 is a
lossless codec, ensuring that every byte of the original file is preserved in
the encoded video without degradation.

## Testing

Once the dependencies are installed you can run the automated tests with:

```
pytest
```

The suite exercises encoding and decoding behavior, including the MP4 fallback
mechanism.
