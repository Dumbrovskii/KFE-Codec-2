import numpy as np

# Simple stub for OpenCV functionality used in tests

def VideoWriter_fourcc(*args):
    return ''.join(args)

class VideoWriter:
    def __init__(self, path, fourcc, fps, frame_size):
        self.path = path
        self.file = open(path, 'wb')
        self.fourcc = fourcc
        self.fps = fps
        self.width, self.height = frame_size
        if fourcc == VideoWriter_fourcc(*"mp4v"):
            # Write minimal MP4-like header so tests can check for 'ftyp'
            self.file.write(b"\x00\x00\x00\x00ftyp")
        else:
            self.file.write(b"KFCD")
        # store width and height for the stub
        self.file.write(self.width.to_bytes(2, 'big'))
        self.file.write(self.height.to_bytes(2, 'big'))
        self.opened = True
    def isOpened(self):
        return self.opened
    def write(self, frame):
        data = np.asarray(frame, dtype=np.uint8).tobytes()
        self.file.write(len(data).to_bytes(4, 'big'))
        self.file.write(data)
    def release(self):
        if self.opened:
            self.file.close()
            self.opened = False

class VideoCapture:
    def __init__(self, path):
        self.path = path
        self.file = open(path, 'rb')
        header = self.file.read(4)
        if header == b"\x00\x00\x00\x00":
            # mp4 header was written; consume 'ftyp'
            self.file.read(4)  # 'ftyp'
        else:
            # assume custom header 'KFCD'
            self.file.read(len('KFCD'))
        self.width = int.from_bytes(self.file.read(2), 'big')
        self.height = int.from_bytes(self.file.read(2), 'big')
        self.frame_size = self.width * self.height * 3
        self.opened = True
    def isOpened(self):
        return self.opened
    def read(self):
        len_bytes = self.file.read(4)
        if not len_bytes:
            return False, None
        length = int.from_bytes(len_bytes, 'big')
        data = self.file.read(length)
        if len(data) < length:
            return False, None
        frame = np.frombuffer(data, dtype=np.uint8).reshape(
            (self.height, self.width, 3)
        )
        return True, frame
    def release(self):
        if self.opened:
            self.file.close()
            self.opened = False
