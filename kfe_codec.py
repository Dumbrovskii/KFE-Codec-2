import argparse
import os
import hashlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cv2

FRAME_WIDTH = 3840
FRAME_HEIGHT = 2160
CHANNELS = 3
BYTES_PER_FRAME = FRAME_WIDTH * FRAME_HEIGHT * CHANNELS


def _chunk_to_frame(chunk: bytes) -> np.ndarray:
    """Convert a BYTES_PER_FRAME sized chunk to a frame array."""
    return np.frombuffer(chunk, dtype=np.uint8).reshape(
        (FRAME_HEIGHT, FRAME_WIDTH, CHANNELS)
    )


def _frame_to_bytes(frame: np.ndarray) -> bytes:
    """Convert a frame array back to raw bytes."""
    return frame.tobytes()


def encode(
    input_path: str,
    output_path: str,
    *,
    container: str = "mkv",
    workers: int = 1,
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
    """

    if not isinstance(workers, int) or workers <= 0:
        raise ValueError("workers must be a positive integer")

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if not output_path.endswith(f".{container}"):
        output_path += f".{container}"

    file_size = os.path.getsize(input_path)

    # Compute checksum in a streaming fashion
    sha = hashlib.sha256()
    with open(input_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(chunk)
    checksum = sha.digest()

    header = (
        file_size.to_bytes(8, "big")
        + checksum
        + b"\x00" * (BYTES_PER_FRAME - 8 - len(checksum))
    )
    header_frame = _chunk_to_frame(header)

    # Select codec based on container
    if container == "mkv":
        fourcc = cv2.VideoWriter_fourcc(*"FFV1")
    elif container == "mp4":
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    else:
        raise ValueError(f"Unsupported container: {container}")

    writer = cv2.VideoWriter(
        output_path, fourcc, 60, (FRAME_WIDTH, FRAME_HEIGHT)
    )
    if not writer.isOpened():
        raise IOError(f"Cannot open video writer for: {output_path}")

    writer.write(header_frame)

    def pad(chunk: bytes) -> bytes:
        if len(chunk) < BYTES_PER_FRAME:
            chunk += b"\x00" * (BYTES_PER_FRAME - len(chunk))
        return chunk

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex, open(
        input_path, "rb"
    ) as f:
        pending = []
        while True:
            chunk = f.read(BYTES_PER_FRAME)
            if not chunk:
                break
            future = ex.submit(_chunk_to_frame, pad(chunk))
            pending.append(future)
            if len(pending) >= workers:
                writer.write(pending.pop(0).result())

        for fut in pending:
            writer.write(fut.result())
    writer.release()


def decode(
    input_path: str, output_path: str, *, workers: int = 1
) -> None:
    """Decode a KFE video back into a binary file."""
    if not isinstance(workers, int) or workers <= 0:
        raise ValueError("workers must be a positive integer")
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        cap.release()
        raise IOError(f"Cannot open video file: {input_path}")

    # Read the header frame containing the original file size
    ret, header_frame = cap.read()
    if not ret:
        cap.release()
        raise IOError("Input video contains no frames")
    header_bytes = header_frame.tobytes()
    original_size = int.from_bytes(header_bytes[:8], "big")
    checksum = header_bytes[8 : 8 + hashlib.sha256().digest_size]

    sha = hashlib.sha256()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex, open(
        output_path, "wb"
    ) as f:
        written = 0
        pending = []
        while True:
            ret, frame = cap.read()
            if not ret or written >= original_size:
                break
            future = ex.submit(_frame_to_bytes, frame)
            pending.append(future)
            if len(pending) >= workers:
                chunk = pending.pop(0).result()
                bytes_to_write = min(original_size - written, len(chunk))
                data = chunk[:bytes_to_write]
                f.write(data)
                sha.update(data)
                written += bytes_to_write

        for fut in pending:
            chunk = fut.result()
            if written >= original_size:
                break
            bytes_to_write = min(original_size - written, len(chunk))
            data = chunk[:bytes_to_write]
            f.write(data)
            sha.update(data)
            written += bytes_to_write
    cap.release()

    if written != original_size or sha.digest() != checksum:
        raise IOError(
            "Video ended before all data could be read or checksum mismatch"
        )


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KFE Codec Prototype")
    subparsers = parser.add_subparsers(dest="command", required=True)

    enc = subparsers.add_parser("encode", help="Encode binary to KFE video")
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
        type=int,
        default=1,
        help="Number of worker threads to use during encoding",
    )

    dec = subparsers.add_parser("decode", help="Decode KFE video to binary")
    dec.add_argument("input_file", help="Path to input video file")
    dec.add_argument("output_file", help="Path to output binary file")
    dec.add_argument(
        "-w",
        "--workers",
        type=int,
        default=1,
        help="Number of worker threads to use during decoding",
    )

    parsed = parser.parse_args(args)
    if hasattr(parsed, "workers") and parsed.workers <= 0:
        raise argparse.ArgumentTypeError("workers must be a positive integer")
    return parsed


def main() -> None:
    args = parse_args()
    if args.command == "encode":
        encode(
            args.input_file,
            args.output_file,
            container=args.container,
            workers=args.workers,
        )
    elif args.command == "decode":
        decode(
            args.input_file,
            args.output_file,
            workers=args.workers,
        )


if __name__ == "__main__":
    main()
