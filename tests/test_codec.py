"""Tests for the KFE codec.

Dependencies for these tests are listed in ``requirements.txt``. Install them with::

    pip install -r requirements.txt
"""

import os
import tempfile

import pytest

from kfe_codec import encode, decode


def test_encode_decode_roundtrip(tmp_path):
    data = os.urandom(1024)
    input_file = tmp_path / 'input.bin'
    video_file = tmp_path / 'output.mp4'
    restored_file = tmp_path / 'restored.bin'

    with open(input_file, 'wb') as f:
        f.write(data)

    encode(str(input_file), str(video_file))
    decode(str(video_file), str(restored_file))

    with open(restored_file, 'rb') as f:
        restored = f.read()

    assert restored == data
