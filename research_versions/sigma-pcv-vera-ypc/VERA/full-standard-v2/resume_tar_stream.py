import json
import re
import zlib

def verified_tar_header(data, member_pattern=r"/Cam[0-5]/[0-9]+[.]tiff$"):
    pos = 32768
    while True:
        marker = data.find(b"ustar", pos)
        if marker < 0:
            return None
        pos = marker + 5
        start = marker - 257
        if start < 32768 or start + 512 > len(data):
            continue
        header = data[start:start + 512]
        try:
            stored = int(header[148:156].strip(bytes([0, 32])) or b"0", 8)
            actual = sum(header[:148]) + 8 * 32 + sum(header[156:])
            name = header[:100].split(bytes([0]), 1)[0].decode("ascii")
        except (ValueError, UnicodeDecodeError):
            continue
        if stored == actual and re.search(member_pattern, name):
            return start, name

def candidate_bit_positions(data):
    limit = min(len(data) - 131073, 512 * 1024)
    for bit in range(max(limit, 0) * 8):
        byte, shift = divmod(bit, 8)
        head = int.from_bytes(data[byte:byte + 4], "little") >> shift
        if head & 7 != 4 or (head >> 3) & 31 > 29 or (head >> 8) & 31 > 29:
            continue
        value = int.from_bytes(data[byte:byte + 10], "little") >> shift
        counts = [0] * 8
        for index in range(4 + ((head >> 13) & 15)):
            length = (value >> (17 + 3 * index)) & 7
            if length:
                counts[length] += 1
        left = 1
        for length in range(1, 8):
            left = left * 2 - counts[length]
            if left < 0:
                break
        if left == 0:
            yield bit

class ResumeTarStream:
    def __init__(self, reader, date):
        self.reader = reader
        self.buffer = bytearray()
        self.ended = False
        for probe in range(8):
            initial_offset = self.reader.initial_position + probe * 4 * 1024 * 1024
            initial = self.reader.read(4 * 1024 * 1024)
            if len(initial) != 4 * 1024 * 1024:
                self.reader.close()
                raise EOFError("Incomplete initial compressed block")
            for bit in candidate_bit_positions(initial):
                byte, shift = divmod(bit, 8)
                raw = initial[byte:]
                held = b""
                if shift:
                    held = raw[-1:]
                    raw = (int.from_bytes(raw, "little") >> shift).to_bytes(len(raw), "little")[:-1]
                try:
                    inflater = zlib.decompressobj(-15, zdict=bytes(32768))
                    decoded = inflater.decompress(raw)
                except zlib.error:
                    continue
                header = verified_tar_header(decoded)
                if header is None:
                    continue
                offset, name = header
                self.bit_offset = (initial_offset + byte) * 8 + shift
                self.first_member = name
                self.shift = shift
                self.held = held
                self.inflater = inflater
                self.buffer.extend(decoded[offset:])
                print("resume_jump_start date={} compressed_start={} bit_offset={} first_member={}".format(
                    date, self.reader.initial_position, self.bit_offset, name), flush=True)
                return
        self.reader.close()
        raise RuntimeError("Could not find verified TAR header within 32 MiB of resume jump start")

    @property
    def position(self):
        return self.reader.position

    def _shift(self, raw):
        if not self.shift:
            return raw
        joined = self.held + raw
        self.held = joined[-1:]
        return (int.from_bytes(joined, "little") >> self.shift).to_bytes(len(joined), "little")[:-1]

    def _pump(self):
        raw = self.reader.read(1024 * 1024)
        if not raw:
            if not self.inflater.eof:
                raise EOFError("Compressed archive ended before DEFLATE end")
            self.ended = True
            return
        try:
            self.buffer.extend(self.inflater.decompress(self._shift(raw)))
        except zlib.error as exc:
            raise RuntimeError("Raw DEFLATE verification failed at compressed offset {}".format(self.position)) from exc
        if self.inflater.eof:
            self.ended = True

    def read(self, size=-1):
        if size < 0:
            raise ValueError("A bounded read size is required")
        while len(self.buffer) < size and not self.ended:
            self._pump()
        result = bytes(self.buffer[:size])
        del self.buffer[:size]
        return result

    def close(self):
        self.reader.close()
