import struct
from typing import List, Tuple


def right_rotate(value: int, shift: int) -> int:
    """Right rotate a 32-bit integer value by shift."""
    return (value >> shift) | (value << (32 - shift)) & 0xFFFFFFFF


def pad_message(message: bytes) -> bytes:
    """Pads the message to be a multiple of 512 bits."""
    original_length = len(message) * 8  # Length of message in bits
    message += b"\x80"  # Append a single '1' bit (as 0x80)
    while (
        len(message) * 8 + 64
    ) % 512 != 0:  # Pad with zeros until length is congruent to 448 (mod 512)
        message += b"\x00"
    message += struct.pack(
        ">Q", original_length
    )  # Append original message length as a 64-bit big-endian integer
    return message


def split_into_blocks(message: bytes) -> List[bytes]:
    """Splits the message into 512-bit blocks."""
    return [message[i : i + 64] for i in range(0, len(message), 64)]


def initialize_hash_values() -> List[int]:
    """Initial hash values defined by the SHA-256 standard."""
    return [
        0x6A09E667,
        0xBB67AE85,
        0x3C6EF372,
        0xA54FF53A,
        0x510E527F,
        0x9B05688C,
        0x1F83D9AB,
        0x5BE0CD19,
    ]


def initialize_constants() -> List[int]:
    """Constants defined by the SHA-256 standard."""
    return [
        0x428A2F98,
        0x71374491,
        0xB5C0FBCF,
        0xE9B5DBA5,
        0x3956C25B,
        0x59F111F1,
        0x923F82A4,
        0xAB1C5ED5,
        0xD807AA98,
        0x12835B01,
        0x243185BE,
        0x550C7DC3,
        0x72BE5D74,
        0x80DEB1FE,
        0x9BDC06A7,
        0xC19BF174,
        0xE49B69C1,
        0xEFBE4786,
        0x0FC19DC6,
        0x240CA1CC,
        0x2DE92C6F,
        0x4A7484AA,
        0x5CB0A9DC,
        0x76F988DA,
        0x983E5152,
        0xA831C66D,
        0xB00327C8,
        0xBF597FC7,
        0xC6E00BF3,
        0xD5A79147,
        0x06CA6351,
        0x14292967,
        0x27B70A85,
        0x2E1B2138,
        0x4D2C6DFC,
        0x53380D13,
        0x650A7354,
        0x766A0ABB,
        0x81C2C92E,
        0x92722C85,
        0xA2BFE8A1,
        0xA81A664B,
        0xC24B8B70,
        0xC76C51A3,
        0xD192E819,
        0xD6990624,
        0xF40E3585,
        0x106AA070,
        0x19A4C116,
        0x1E376C08,
        0x2748774C,
        0x34B0BCB5,
        0x391C0CB3,
        0x4ED8AA4A,
        0x5B9CCA4F,
        0x682E6FF3,
        0x748F82EE,
        0x78A5636F,
        0x84C87814,
        0x8CC70208,
        0x90BEFFFA,
        0xA4506CEB,
        0xBEF9A3F7,
        0xC67178F2,
    ]


def process_block(block: bytes, H: List[int], K: List[int]) -> List[int]:
    """Process a single 512-bit block."""
    w = (
        list(struct.unpack(">16L", block)) + [0] * 48
    )  # Unpack 512-bit block into sixteen 32-bit big-endian words
    for j in range(16, 64):
        s0 = right_rotate(w[j - 15], 7) ^ right_rotate(w[j - 15], 18) ^ (w[j - 15] >> 3)
        s1 = right_rotate(w[j - 2], 17) ^ right_rotate(w[j - 2], 19) ^ (w[j - 2] >> 10)
        w[j] = (w[j - 16] + s0 + w[j - 7] + s1) & 0xFFFFFFFF  # Word expansion

    a, b, c, d, e, f, g, h = H

    for j in range(64):
        S1 = right_rotate(e, 6) ^ right_rotate(e, 11) ^ right_rotate(e, 25)
        ch = (e & f) ^ (~e & g)
        temp1 = (h + S1 + ch + K[j] + w[j]) & 0xFFFFFFFF
        S0 = right_rotate(a, 2) ^ right_rotate(a, 13) ^ right_rotate(a, 22)
        maj = (a & b) ^ (a & c) ^ (b & c)
        temp2 = (S0 + maj) & 0xFFFFFFFF

        h = g
        g = f
        f = e
        e = (d + temp1) & 0xFFFFFFFF
        d = c
        c = b
        b = a
        a = (temp1 + temp2) & 0xFFFFFFFF

    return [(x + y) & 0xFFFFFFFF for x, y in zip(H, [a, b, c, d, e, f, g, h])]


def sha256(message: bytes) -> str:
    """SHA-256 hashing algorithm."""
    message = pad_message(message)
    blocks = split_into_blocks(message)
    H = initialize_hash_values()
    K = initialize_constants()

    for block in blocks:
        H = process_block(block, H, K)

    return "".join(f"{x:08x}" for x in H)


message = b"Hello, message"
print(f"Original Text: '{message}'")
print(f"SHA-256 Text: '{sha256(message)}'")

