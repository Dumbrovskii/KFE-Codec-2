import argparse
import os
import numpy as np
import cv2

FRAME_WIDTH = 3840
FRAME_HEIGHT = 2160
CHANNELS = 3
BYTES_PER_FRAME = FRAME_WIDTH * FRAME_HEIGHT * CHANNELS


def encode(input_path: str, output_path: str) -> None:
    """Encode a binary file into a KFE video."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(input_path, 'rb') as f:
        data = f.read()

    # First frame stores the original file size so decoding can trim padding
    header = len(data).to_bytes(8, 'big') + b'\x00' * (BYTES_PER_FRAME - 8)
    header_frame = np.frombuffer(header, dtype=np.uint8).reshape((FRAME_HEIGHT, FRAME_WIDTH, CHANNELS))

    # Use a lossless codec so encoded videos preserve the exact binary data.
    # FFV1 is a widely supported lossless codec available in FFMPEG builds
    # shipped with OpenCV.
    fourcc = cv2.VideoWriter_fourcc(*'FFV1')
    writer = cv2.VideoWriter(output_path, fourcc, 60, (FRAME_WIDTH, FRAME_HEIGHT))

    writer.write(header_frame)

    for i in range(0, len(data), BYTES_PER_FRAME):
        chunk = data[i:i + BYTES_PER_FRAME]
        if len(chunk) < BYTES_PER_FRAME:
            chunk += b'\x00' * (BYTES_PER_FRAME - len(chunk))
        frame = np.frombuffer(chunk, dtype=np.uint8).reshape((FRAME_HEIGHT, FRAME_WIDTH, CHANNELS))
        writer.write(frame)
    writer.release()


def decode(input_path: str, output_path: str) -> None:
    """Decode a KFE video back into a binary file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {input_path}")

    # Read the header frame containing the original file size
    ret, header_frame = cap.read()
    if not ret:
        cap.release()
        raise IOError("Input video contains no frames")
    header_bytes = header_frame.tobytes()
    original_size = int.from_bytes(header_bytes[:8], 'big')

    binary_chunks = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        binary_chunks.append(frame.tobytes())
    cap.release()

    if binary_chunks:
        binary_data = b''.join(binary_chunks)
        binary_data = binary_data[:original_size]
    else:
        binary_data = b''

    with open(output_path, 'wb') as f:
        f.write(binary_data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KFE Codec Prototype")
    subparsers = parser.add_subparsers(dest="command", required=True)

    enc = subparsers.add_parser('encode', help='Encode binary to KFE video')
    enc.add_argument('input_file', help='Path to input binary file')
    enc.add_argument('output_file', help='Path to output video file')

    dec = subparsers.add_parser('decode', help='Decode KFE video to binary')
    dec.add_argument('input_file', help='Path to input video file')
    dec.add_argument('output_file', help='Path to output binary file')

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == 'encode':
        encode(args.input_file, args.output_file)
    elif args.command == 'decode':
        decode(args.input_file, args.output_file)


if __name__ == '__main__':
    main()
