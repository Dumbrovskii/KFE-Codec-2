"""Tests for the KFE codec.

Dependencies for these tests are listed in ``requirements.txt``. Install them with::

    pip install -r requirements.txt
"""

import os
import sys

import pytest
import cv2
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from kfe_codec import encode, decode, BYTES_PER_FRAME, parse_args
import argparse


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
    video_file = tmp_path / 'empty.mkv'
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
    video_file = tmp_path / 'large.mkv'
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
    video_file = tmp_path / 'out.mkv'
    with pytest.raises(FileNotFoundError):
        encode(str(missing_file), str(video_file))


def test_decode_missing_input(tmp_path):
    missing_video = tmp_path / 'nope.mkv'
    output_file = tmp_path / 'out.bin'
    with pytest.raises(IOError):
        decode(str(missing_video), str(output_file))


def test_decode_missing_input_releases(tmp_path, monkeypatch):
    mock_cap = mock.Mock()
    mock_cap.isOpened.return_value = False

    def fake_video_capture(path):
        return mock_cap

    monkeypatch.setattr(cv2, 'VideoCapture', fake_video_capture)

    with pytest.raises(IOError):
        decode('badpath.mkv', str(tmp_path / 'out.bin'))

    mock_cap.release.assert_called_once()


def test_decode_truncated_video(tmp_path):
    data = os.urandom(100)
    input_file = tmp_path / 'input.bin'
    full_video = tmp_path / 'full.mkv'
    truncated_video = tmp_path / 'truncated.mkv'
    output_file = tmp_path / 'out.bin'

    with open(input_file, 'wb') as f:
        f.write(data)

    encode(str(input_file), str(full_video))

    # Create a truncated copy of the video file
    with open(full_video, 'rb') as src, open(truncated_video, 'wb') as dst:
        content = src.read()
        dst.write(content[: len(content) // 2])

    with pytest.raises(IOError):
        decode(str(truncated_video), str(output_file))


def test_decode_random_mp4_fails(tmp_path):
    """Decoding a non-KFE MP4 should raise an IOError."""
    random_mp4 = tmp_path / "random.mp4"
    output = tmp_path / "out.bin"

    # Write some random bytes so the file exists but is not a valid video
    random_mp4.write_bytes(os.urandom(1024))

    with pytest.raises(IOError):
        decode(str(random_mp4), str(output))


def test_decode_mkv_with_mp4_extension(tmp_path):
    """Decoding should work even if an MKV file has a .mp4 extension."""
    data = os.urandom(256)
    input_file = tmp_path / "input.bin"
    mkv_file = tmp_path / "video.mkv"
    mp4_alias = tmp_path / "video.mp4"
    restored = tmp_path / "restored.bin"

    with open(input_file, "wb") as f:
        f.write(data)

    encode(str(input_file), str(mkv_file))
    mkv_file.rename(mp4_alias)

    decode(str(mp4_alias), str(restored))

    with open(restored, "rb") as f:
        output_data = f.read()

    assert output_data == data


def _file_hash(path):
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def test_encode_decode_multi_frame(tmp_path):
    size = BYTES_PER_FRAME * 3 + 1234
    input_file = tmp_path / "multi.bin"
    video_file = tmp_path / "multi.mkv"
    restored_file = tmp_path / "restored.bin"

    # Write random bytes without holding the entire payload in memory
    remaining = size
    with open(input_file, "wb") as f:
        while remaining > 0:
            chunk = os.urandom(min(1024 * 1024, remaining))
            f.write(chunk)
            remaining -= len(chunk)

    encode(str(input_file), str(video_file))
    decode(str(video_file), str(restored_file))

    assert _file_hash(input_file) == _file_hash(restored_file)


def test_mp4_roundtrip_workers(tmp_path):
    data = os.urandom(4096)
    input_file = tmp_path / "input.bin"
    video_file = tmp_path / "video.mp4"
    restored_file = tmp_path / "restored.bin"

    with open(input_file, "wb") as f:
        f.write(data)

    encode(str(input_file), str(video_file), container="mp4", workers=2)
    decode(str(video_file), str(restored_file), workers=2)

    assert _file_hash(input_file) == _file_hash(restored_file)


def test_checksum_mismatch(tmp_path):
    data = os.urandom(2048)
    input_file = tmp_path / "input.bin"
    video_file = tmp_path / "video.mkv"
    output_file = tmp_path / "out.bin"

    with open(input_file, "wb") as f:
        f.write(data)

    encode(str(input_file), str(video_file))

    # Corrupt a byte somewhere in the middle of the video
    size = os.path.getsize(video_file)
    with open(video_file, "r+b") as f:
        f.seek(size // 2)
        b = f.read(1)
        if not b:
            pytest.skip("Could not corrupt file: no byte read")
        f.seek(-1, os.SEEK_CUR)
        f.write(b"\x00" if b != b"\x00" else b"\x01")

    with pytest.raises(IOError):
        decode(str(video_file), str(output_file))


def test_encode_invalid_workers(tmp_path):
    input_file = tmp_path / "in.bin"
    with open(input_file, "wb") as f:
        f.write(b"data")

    with pytest.raises(ValueError):
        encode(str(input_file), str(tmp_path / "out.mkv"), workers=0)


def test_decode_invalid_workers(tmp_path):
    data = b"hello"
    input_file = tmp_path / "input.bin"
    video_file = tmp_path / "video.mkv"
    with open(input_file, "wb") as f:
        f.write(data)

    encode(str(input_file), str(video_file))

    with pytest.raises(ValueError):
        decode(str(video_file), str(tmp_path / "out.bin"), workers=-1)


def test_parse_args_invalid_workers():
    with pytest.raises(argparse.ArgumentTypeError):
        parse_args(["encode", "in.bin", "out.mkv", "--workers", "0"])
