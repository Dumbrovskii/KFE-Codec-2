import os
import hashlib
from typing import Callable

import cv2

from kfe_codec import (
    FRAME_WIDTH,
    FRAME_HEIGHT,
    CHANNELS,
    BYTES_PER_FRAME,
    _chunk_to_frame,
    _frame_to_bytes,
    _derive_vk3,
    video_writer,
    video_capture,
)


def loopback(
    input_path: str,
    output_path: str,
    *,
    certificate: str | None = None,
    progress: bool = False,
    writer_factory: Callable[..., cv2.VideoWriter] = cv2.VideoWriter,
    capture_factory: Callable[..., cv2.VideoCapture] = cv2.VideoCapture,
) -> None:
    """Encode ``input_path`` and transmit via HDMI loop-back to ``output_path``.

    Both encoding and decoding happen in real-time using the provided writer and
    capture factories which default to ``cv2.VideoWriter`` and
    ``cv2.VideoCapture``. The function assumes the HDMI output is connected to
    the HDMI input so that frames written by the writer can be read by the
    capture.
    """

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if certificate is None:
        default_cert = output_path + ".cert"
        if os.path.exists(default_cert):
            certificate = default_cert

    # Pre-compute file size and checksum
    sha = hashlib.sha256()
    file_size = 0
    with open(input_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            file_size += len(chunk)
            sha.update(chunk)
    checksum = sha.digest()

    cert_checksum = b"\x00" * hashlib.sha256().digest_size
    vk3 = None
    if certificate:
        with open(certificate, "rb") as cf:
            cert_bytes = cf.read()
        if not cert_bytes:
            raise ValueError("Certificate file is empty")
        cert_checksum = hashlib.sha256(cert_bytes).digest()
        vk3 = _derive_vk3(cert_bytes)
        out_cert_path = output_path + ".cert"
        if certificate != out_cert_path:
            import shutil

            shutil.copyfile(certificate, out_cert_path)

    def pad(chunk: bytes) -> bytes:
        if len(chunk) < BYTES_PER_FRAME:
            chunk += b"\x00" * (BYTES_PER_FRAME - len(chunk))
        return chunk

    header = (
        file_size.to_bytes(8, "big")
        + checksum
        + cert_checksum
        + b"\x00" * (BYTES_PER_FRAME - 8 - len(checksum) - len(cert_checksum))
    )
    header_frame = _chunk_to_frame(header)

    fourcc = cv2.VideoWriter_fourcc(*"FFV1")
    with video_writer(
        0, fourcc, 60, (FRAME_WIDTH, FRAME_HEIGHT), factory=writer_factory
    ) as writer, video_capture(0, factory=capture_factory) as cap:
        if not writer.isOpened() or not cap.isOpened():
            raise IOError("Could not open HDMI devices")

        writer.write(header_frame)
        ret, rx_header = cap.read()
        if not ret:
            raise IOError("Failed to read header frame from HDMI input")

        header_bytes = rx_header.tobytes()
        digest_len = hashlib.sha256().digest_size
        original_size = int.from_bytes(header_bytes[:8], "big")
        checksum_rx = header_bytes[8 : 8 + digest_len]
        cert_checksum_stored = header_bytes[8 + digest_len : 8 + 2 * digest_len]

        if original_size != file_size or checksum_rx != checksum:
            raise IOError("Header mismatch on loop-back")

        if certificate:
            if cert_checksum != cert_checksum_stored:
                raise IOError("Certificate checksum mismatch")
        elif any(cert_checksum_stored):
            raise IOError("Certificate required for decoding")

        sha_out = hashlib.sha256()
        with open(input_path, "rb") as fin, open(output_path, "wb") as fout:
            written = 0
            frame_total = (file_size + BYTES_PER_FRAME - 1) // BYTES_PER_FRAME
            processed = 0
            while written < file_size:
                chunk = fin.read(BYTES_PER_FRAME)
                if not chunk:
                    break
                frame = _chunk_to_frame(pad(chunk), vk3)
                writer.write(frame)

                ret, rx_frame = cap.read()
                if not ret:
                    raise IOError("Lost frame on HDMI input")
                data = _frame_to_bytes(rx_frame, vk3)
                bytes_to_write = min(file_size - written, len(data))
                portion = data[:bytes_to_write]
                fout.write(portion)
                sha_out.update(portion)
                written += bytes_to_write
                processed += 1
                if progress:
                    print(
                        f"Loopback {processed}/{frame_total} frames",
                        end="\r",
                        flush=True,
                    )
            if progress:
                print()

    if sha_out.digest() != checksum:
        raise IOError("Checksum mismatch after loop-back")
