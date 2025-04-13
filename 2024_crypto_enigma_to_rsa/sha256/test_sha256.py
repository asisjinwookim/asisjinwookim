import struct
from hashlib import sha256 as hashlib_sha256

from .sha256 import (
    right_rotate,
    pad_message,
    split_into_blocks,
    initialize_hash_values,
    initialize_constants,
    process_block,
    sha256,
)


def test_right_rotate():
    assert (
        right_rotate(0b00000000000000000000000000000001, 1)
        == 0b10000000000000000000000000000000
    )
    assert (
        right_rotate(0b10000000000000000000000000000000, 1)
        == 0b01000000000000000000000000000000
    )
    assert right_rotate(0x12345678, 4) == 0x81234567


def test_pad_message():
    message = b"abc"
    padded_message = pad_message(message)
    assert len(padded_message) % 64 == 0
    assert padded_message[-1] == 0x18  # Length of "abc" in bits is 24 (0x18)


def test_split_into_blocks():
    message = b"abc"
    padded_message = pad_message(message)
    blocks = split_into_blocks(padded_message)
    assert len(blocks) == 1
    block = blocks[0]
    assert len(block) == 64
    assert block.startswith(b"abc\x80")
    assert block.count(b"\x00") == 59
    assert block.endswith(struct.pack(">Q", 24))
    assert block == b"abc\x80" + b"\x00" * 59 + b"\x18"


def test_initialize_hash_values():
    H = initialize_hash_values()
    assert len(H) == 8
    expected_values = [
        0x6A09E667,
        0xBB67AE85,
        0x3C6EF372,
        0xA54FF53A,
        0x510E527F,
        0x9B05688C,
        0x1F83D9AB,
        0x5BE0CD19,
    ]
    assert H == expected_values


def test_initialize_constants():
    K = initialize_constants()
    assert len(K) == 64
    expected_first_values = [
        0x428A2F98,
        0x71374491,
        0xB5C0FBCF,
        0xE9B5DBA5,
        0x3956C25B,
        0x59F111F1,
        0x923F82A4,
        0xAB1C5ED5,
    ]
    assert K[:8] == expected_first_values


def test_process_block():
    H = initialize_hash_values()
    K = initialize_constants()
    message = b"abc" + b"\x80" + b"\x00" * 60 + struct.pack(">Q", 24)
    block = message[:64]
    new_H = process_block(block, H, K)
    assert len(new_H) == 8


def test_sha256():
    message = b"abc"
    computed_hash = sha256(message)
    expected_hash = hashlib_sha256(message).hexdigest()
    assert computed_hash == expected_hash
