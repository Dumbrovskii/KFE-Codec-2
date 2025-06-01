import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kfe_loopback import loopback
from kfe_codec import BYTES_PER_FRAME


class _FakeDevice:
    def __init__(self):
        self.frames = []
        self.read_idx = 0

    class Writer:
        def __init__(self, parent):
            self.parent = parent

        def isOpened(self):
            return True

        def write(self, frame):
            self.parent.frames.append(frame.copy())

        def release(self):
            pass

    class Capture:
        def __init__(self, parent):
            self.parent = parent

        def isOpened(self):
            return True

        def read(self):
            if self.parent.read_idx < len(self.parent.frames):
                f = self.parent.frames[self.parent.read_idx]
                self.parent.read_idx += 1
                return True, f.copy()
            return False, None

        def release(self):
            pass

    def writer_factory(self, *args, **kwargs):
        return self.Writer(self)

    def capture_factory(self, *args, **kwargs):
        return self.Capture(self)


def test_loopback_roundtrip(tmp_path):
    device = _FakeDevice()
    data = os.urandom(BYTES_PER_FRAME + 123)
    inp = tmp_path / "in.bin"
    out = tmp_path / "out.bin"

    inp.write_bytes(data)

    loopback(
        str(inp),
        str(out),
        writer_factory=device.writer_factory,
        capture_factory=device.capture_factory,
    )

    assert out.read_bytes() == data


def test_loopback_with_certificate(tmp_path):
    device = _FakeDevice()
    data = os.urandom(BYTES_PER_FRAME + 50)
    inp = tmp_path / "in.bin"
    out = tmp_path / "out.bin"
    cert = tmp_path / "enc.cert"

    inp.write_bytes(data)
    cert.write_bytes(b"cert-data")

    loopback(
        str(inp),
        str(out),
        certificate=str(cert),
        writer_factory=device.writer_factory,
        capture_factory=device.capture_factory,
    )

    assert out.read_bytes() == data
    out_cert = tmp_path / "out.bin.cert"
    assert out_cert.exists()
    assert out_cert.read_bytes() == b"cert-data"


def test_loopback_default_certificate(tmp_path):
    device = _FakeDevice()
    data = os.urandom(BYTES_PER_FRAME)
    inp = tmp_path / "input.bin"
    out = tmp_path / "output.bin"
    default_cert = tmp_path / "output.bin.cert"

    inp.write_bytes(data)
    default_cert.write_bytes(b"default-cert")

    loopback(
        str(inp),
        str(out),
        writer_factory=device.writer_factory,
        capture_factory=device.capture_factory,
    )

    assert out.read_bytes() == data
    assert default_cert.read_bytes() == b"default-cert"
