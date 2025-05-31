"""Tests for the KFE codec.

Dependencies for these tests are listed in ``requirements.txt``. Install them with::

    pip install -r requirements.txt
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from kfe_codec import encode, decode, BYTES_PER_FRAME


def test_encode_decode_roundtrip(tmp_path):
    data = os.urandom(1024)
    input_file = tmp_path / 'input.bin'
    video_file = tmp_path / 'output.mkv'
    restored_file = tmp_path / 'restored.bin'

    with open(input_file, 'wb') as f:
        f.write(data)

    encode(str(input_file), str(video_file))
    decode(str(video_file), str(restored_file))

    with open(restored_file, 'rb') as f:
        restored = f.read()

    assert restored == data


def test_encode_decode_empty_file(tmp_path):
    input_file = tmp_path / 'empty.bin'
    video_file = tmp_path / 'empty.mp4'
    restored_file = tmp_path / 'restored.bin'

    open(input_file, 'wb').close()

    encode(str(input_file), str(video_file))
    decode(str(video_file), str(restored_file))

    with open(restored_file, 'rb') as f:
        restored = f.read()

    assert restored == b''


def test_encode_decode_large_file(tmp_path):
    data = os.urandom(BYTES_PER_FRAME + 500)
    input_file = tmp_path / 'large.bin'
    video_file = tmp_path / 'large.mp4'
    restored_file = tmp_path / 'restored.bin'

    with open(input_file, 'wb') as f:
        f.write(data)

    encode(str(input_file), str(video_file))
    decode(str(video_file), str(restored_file))

    with open(restored_file, 'rb') as f:
        restored = f.read()

    assert restored == data


def test_encode_missing_input(tmp_path):
    missing_file = tmp_path / 'nope.bin'
    video_file = tmp_path / 'out.mp4'
    with pytest.raises(FileNotFoundError):
        encode(str(missing_file), str(video_file))


def test_decode_missing_input(tmp_path):
    missing_video = tmp_path / 'nope.mp4'
    output_file = tmp_path / 'out.bin'
    with pytest.raises(IOError):
        decode(str(missing_video), str(output_file))
