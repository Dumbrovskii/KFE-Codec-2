import argparse
import os
import hashlib
import math
import shutil


try:
    from numba import njit
except Exception:  # pragma: no cover - numba may be unavailable
    def njit(*args, **kwargs):
        def wrapper(func):
            return func
        return wrapper

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import numpy as np
import cv2

FRAME_WIDTH = 3840
FRAME_HEIGHT = 2160
CHANNELS = 3
BYTES_PER_FRAME = FRAME_WIDTH * FRAME_HEIGHT * CHANNELS

# Cached forward and inverse cpECSK permutation indices
_PERM_CACHE: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}


@njit(cache=True)
def _numba_coprime_shift(arr: np.ndarray, a: int) -> np.ndarray:
    """Return a copy of ``arr`` with channels permuted by coprime ``a``."""
    n_rows, n_cols = arr.shape
    result = np.empty_like(arr)
    for j in range(n_cols):
        new_j = (j * a) % n_cols
        for i in range(n_rows):
            result[i, new_j] = arr[i, j]
    return result


@contextmanager
def video_writer(*args, factory=None, **kwargs):
    """Context manager that releases ``cv2.VideoWriter``."""
    if factory is None:
        factory = cv2.VideoWriter
    writer = factory(*args, **kwargs)
    try:
        yield writer
    finally:
        writer.release()


@contextmanager
def video_capture(*args, factory=None, **kwargs):
    """Context manager that releases ``cv2.VideoCapture``."""
    if factory is None:
        factory = cv2.VideoCapture
    cap = factory(*args, **kwargs)
    try:
        yield cap
    finally:
        cap.release()


def positive_int(value: str) -> int:
    """Parse ``value`` into a positive integer for argparse."""
    try:
        ivalue = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a valid integer")
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("workers must be a positive integer")
    return ivalue


def _chunk_to_frame(chunk: bytes, vk3: tuple[int, int, int] | None = None) -> np.ndarray:

    """Convert a BYTES_PER_FRAME sized chunk to a frame array.

    If ``vk3`` is provided, apply cpECSK permutation.
    """
    frame = np.frombuffer(chunk, dtype=np.uint8).reshape(
        (FRAME_HEIGHT, FRAME_WIDTH, CHANNELS)
    )
    if vk3 is not None:
        frame = _cpECSK_permute(frame, vk3, encode=True)
    return frame


def _frame_to_bytes(frame: np.ndarray, vk3: tuple[int, int, int] | None = None) -> bytes:
    """Convert a frame array back to raw bytes.

    If ``vk3`` is provided, reverse the cpECSK permutation first.
    """
    if vk3 is not None:
        frame = _cpECSK_permute(frame, vk3, encode=False)
    return frame.tobytes()



def _cpECSK_permute(
    frame: np.ndarray, vk3: tuple[int, int, int], *, encode: bool
) -> np.ndarray:
    """Apply or reverse cpECSK pixel permutation on ``frame`` using cached indices."""
    a, b, channel_a = vk3
    n_pixels = FRAME_WIDTH * FRAME_HEIGHT
    flat = frame.reshape(n_pixels, CHANNELS)

    key = (a, b)
    perms = _PERM_CACHE.get(key)
    if perms is None:
        fwd = (np.arange(n_pixels) * a + b) % n_pixels
        inv = (pow(a, -1, n_pixels) * (np.arange(n_pixels) - b)) % n_pixels
        perms = (fwd.astype(np.int64), inv.astype(np.int64))
        _PERM_CACHE[key] = perms
    fwd_idx, inv_idx = perms

    if encode:
        flat = flat[fwd_idx]
        if channel_a != 1:
            flat = _numba_coprime_shift(flat, channel_a)
    else:
        if channel_a != 1:
            flat = _numba_coprime_shift(flat, channel_a)
        flat = flat[inv_idx]

    return flat.reshape(FRAME_HEIGHT, FRAME_WIDTH, CHANNELS)


def _derive_vk3(cert_bytes: bytes) -> tuple[int, int, int]:
    """Derive the cpECSK key tuple ``(a, b, channel_a)`` from certificate bytes."""
    n_pixels = FRAME_WIDTH * FRAME_HEIGHT
    digest = hashlib.sha256(cert_bytes).digest()
    seed = int.from_bytes(digest[:4], "big") % n_pixels
    a = seed or 1
    while math.gcd(a, n_pixels) != 1:
        a = (a + 1) % n_pixels or 1
    b = cert_bytes[0]
    channel_a = (b % (CHANNELS - 1)) + 1
    return a, b, channel_a


def encode(
    input_path: str,
    output_path: str,
    *,
    container: str = "mkv",
    workers: int = 1,
    progress: bool = False,
    certificate: str | None = None,
) -> None:
    """Encode a binary file into a KFE video.

    Parameters
    ----------
    input_path:
        Path to the binary file to encode.
    output_path:
        Path to the output video file. The extension is optional and is
        determined from ``container`` if missing.
    container:
        Either ``"mkv"`` (lossless FFV1) or ``"mp4"`` (MPEG-4). Defaults to
        ``"mkv"``.
    workers:
        Number of worker threads used when converting chunks to frames. The
        writer itself remains sequential.
    progress:
        If ``True``, display progress information during encoding.
    certificate:
        Optional path to a certificate file used for cpECSK encryption.
    """

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if not output_path.endswith(f".{container}"):
        output_path += f".{container}"

    # First pass: compute checksum and file size
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

    # Select codec based on container
    if container == "mkv":
        fourcc = cv2.VideoWriter_fourcc(*"FFV1")
        write_path = output_path
    elif container == "mp4":
        # Use a standard MPEG-4 codec and write directly to MP4
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        write_path = output_path
    else:
        raise ValueError(f"Unsupported container: {container}")

    with video_writer(write_path, fourcc, 60, (FRAME_WIDTH, FRAME_HEIGHT)) as writer:
        if not writer.isOpened():
            raise IOError(f"Cannot open video writer for: {output_path}")

        writer.write(header_frame)

        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex, open(
            input_path, "rb"
        ) as f:
            pending = []
            frame_total = (file_size + BYTES_PER_FRAME - 1) // BYTES_PER_FRAME
            written_frames = 0
            while True:
                chunk = f.read(BYTES_PER_FRAME)
                if not chunk:
                    break
                future = ex.submit(_chunk_to_frame, pad(chunk), vk3)
                pending.append(future)
                if len(pending) >= workers:
                    writer.write(pending.pop(0).result())
                    written_frames += 1
                    if progress:
                        print(
                            f"Encoding {written_frames}/{frame_total} frames",
                            end="\r",
                            flush=True,
                        )

            for fut in pending:
                writer.write(fut.result())
                written_frames += 1
                if progress:
                    print(
                        f"Encoding {written_frames}/{frame_total} frames",
                        end="\r",
                        flush=True,
                    )
            if progress:
                print()



def decode(
    input_path: str,
    output_path: str,
    *,
    workers: int = 1,
    progress: bool = False,
    certificate: str | None = None,
) -> None:
    """Decode a KFE video back into a binary file.

    Parameters
    ----------
    input_path:
        Path to the video file to decode.
    output_path:
        Path where the restored binary data will be written.
    workers:
        Number of worker threads used when converting frames back to bytes.
    progress:
        If ``True``, display progress information during decoding.
    certificate:
        Optional path to the certificate file required for cpECSK decoding.
    """
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if certificate is None:
        default_cert = input_path + ".cert"
        if os.path.exists(default_cert):
            certificate = default_cert

    with video_capture(input_path) as cap:
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {input_path}")

        # Read the header frame containing the original file size
        ret, header_frame = cap.read()
        if not ret:
            raise IOError("Input video contains no frames")
        header_bytes = header_frame.tobytes()
        digest_len = hashlib.sha256().digest_size
        original_size = int.from_bytes(header_bytes[:8], "big")
        checksum = header_bytes[8 : 8 + digest_len]
        cert_checksum_stored = header_bytes[8 + digest_len : 8 + 2 * digest_len]

        vk3 = None
        if certificate:
            with open(certificate, "rb") as cf:
                cert_bytes = cf.read()
            if not cert_bytes:
                raise IOError("Certificate file is empty")
            cert_checksum = hashlib.sha256(cert_bytes).digest()
            if cert_checksum != cert_checksum_stored:
                raise IOError("Certificate checksum mismatch")
            vk3 = _derive_vk3(cert_bytes)
        elif any(cert_checksum_stored):
            raise IOError("Certificate required for decoding")

        sha = hashlib.sha256()
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex, open(
            output_path, "wb"
        ) as f:
            written = 0
            pending = []
            frame_total = (original_size + BYTES_PER_FRAME - 1) // BYTES_PER_FRAME
            decoded_frames = 0
            while True:
                ret, frame = cap.read()
                if not ret or written >= original_size:
                    break
                future = ex.submit(_frame_to_bytes, frame, vk3)
                pending.append(future)
                if len(pending) >= workers:
                    chunk = pending.pop(0).result()
                    bytes_to_write = min(original_size - written, len(chunk))
                    data = chunk[:bytes_to_write]
                    f.write(data)
                    sha.update(data)
                    written += bytes_to_write
                    decoded_frames += 1
                    if progress:
                        print(
                            f"Decoding {decoded_frames}/{frame_total} frames",
                            end="\r",
                            flush=True,
                        )

            for fut in pending:
                chunk = fut.result()
                if written >= original_size:
                    break
                bytes_to_write = min(original_size - written, len(chunk))
                data = chunk[:bytes_to_write]
                f.write(data)
                sha.update(data)
                written += bytes_to_write
                decoded_frames += 1
                if progress:
                    print(
                        f"Decoding {decoded_frames}/{frame_total} frames",
                        end="\r",
                        flush=True,
                    )
            if progress:
                print()

    if written != original_size or sha.digest() != checksum:
        raise IOError("Video ended before all data could be read or checksum mismatch")


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KFE Codec Prototype", exit_on_error=False
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    enc = subparsers.add_parser(
        "encode", help="Encode binary to KFE video", exit_on_error=False
    )
    enc.add_argument("input_file", help="Path to input binary file")
    enc.add_argument("output_file", help="Path to output video file")
    enc.add_argument(
        "-c",
        "--container",
        choices=["mkv", "mp4"],
        default="mkv",
        help="Container format for the output video",
    )
    enc.add_argument(
        "-w",
        "--workers",
        type=positive_int,
        default=1,
        help="Number of worker threads to use during encoding",
    )
    enc.add_argument(
        "--progress",
        action="store_true",
        help="Show progress information while encoding",
    )
    enc.add_argument(
        "--cert",
        dest="certificate",
        help="Path to certificate file for cpECSK encryption",
    )

    dec = subparsers.add_parser(
        "decode", help="Decode KFE video to binary", exit_on_error=False
    )
    dec.add_argument("input_file", help="Path to input video file")
    dec.add_argument("output_file", help="Path to output binary file")
    dec.add_argument(
        "-w",
        "--workers",
        type=positive_int,
        default=1,
        help="Number of worker threads to use during decoding",
    )
    dec.add_argument(
        "--progress",
        action="store_true",
        help="Show progress information while decoding",
    )
    dec.add_argument(
        "--cert",
        dest="certificate",
        help="Path to certificate file for cpECSK decoding",
    )

    loop = subparsers.add_parser(
        "loopback", help="Encode and decode via HDMI loop-back", exit_on_error=False
    )
    loop.add_argument("input_file", help="Path to input binary file")
    loop.add_argument("output_file", help="Path to output binary file")
    loop.add_argument(
        "--progress",
        action="store_true",
        help="Show progress information during loop-back",
    )
    loop.add_argument(
        "--cert",
        dest="certificate",
        help="Path to certificate file for cpECSK",
    )
    try:
        return parser.parse_args(args)
    except argparse.ArgumentError as err:
        raise argparse.ArgumentTypeError(str(err))


def main() -> None:
    args = parse_args()
    if args.command == "encode":
        encode(
            args.input_file,
            args.output_file,
            container=args.container,
            workers=args.workers,
            progress=args.progress,
            certificate=args.certificate,
        )
    elif args.command == "decode":
        decode(
            args.input_file,
            args.output_file,
            workers=args.workers,
            progress=args.progress,
            certificate=args.certificate,
        )
    elif args.command == "loopback":
        from kfe_loopback import loopback

        loopback(
            args.input_file,
            args.output_file,
            progress=args.progress,
            certificate=args.certificate,
        )


if __name__ == "__main__":
    main()
