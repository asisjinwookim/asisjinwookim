

from typing import List
from rich import print
from rich.table import Table

# Simple S-box for substitution (just an example, not cryptographically secure)
sbox = [
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
]

# Reverse S-box for decryption
inv_sbox = [sbox.index(x) for x in range(256)]

class AES16:
    def __init__(self, key: str):
        if len(key) != 16:
            raise ValueError("Key must be 16 characters long.")
        self.key = [ord(char) for char in key]

    def sub_bytes(self, state: List[List[int]], sbox: List[int]) -> List[List[int]]:
        """Substitute bytes in the state matrix using the S-box"""
        state = [[sbox[byte] for byte in row] for row in state]
        self.print_state(state, "SubBytes")
        return state

    def shift_rows(self, state: List[List[int]], inverse: bool = False) -> List[List[int]]:
        """Shift rows of the state matrix"""
        if inverse:
            state[1] = state[1][-1:] + state[1][:-1]  # Rotate second row right
            state[2] = state[2][-2:] + state[2][:-2]  # Rotate third row right by 2
            state[3] = state[3][-3:] + state[3][:-3]  # Rotate fourth row right by 3
        else:
            state[1] = state[1][1:] + state[1][:1]  # Rotate second row left
            state[2] = state[2][2:] + state[2][:2]  # Rotate third row left by 2
            state[3] = state[3][3:] + state[3][:3]  # Rotate fourth row left by 3
        self.print_state(state, "ShiftRows")
        return state

    def add_round_key(self, state: List[List[int]], key: List[List[int]]) -> List[List[int]]:
        """XOR the state matrix with the key matrix"""
        state = [[state[i][j] ^ key[i][j] for j in range(4)] for i in range(4)]
        self.print_state(state, "AddRoundKey")
        return state

    def encrypt(self, plaintext: str) -> List[int]:
        """Encrypt the plaintext"""
        if len(plaintext) != 16:
            raise ValueError("Plaintext must be 16 characters long.")
        state = [[ord(plaintext[i * 4 + j]) for j in range(4)] for i in range(4)]
        key_matrix = [[self.key[i * 4 + j] for j in range(4)] for i in range(4)]

        print("[bold]Initial State[/bold]")
        self.print_state(state, "Initial State")
        self.print_state(key_matrix, "Key Matrix")

        state = self.add_round_key(state, key_matrix)
        state = self.sub_bytes(state, sbox)
        state = self.shift_rows(state)
        state = self.add_round_key(state, key_matrix)

        encrypted = [byte for row in state for byte in row]
        print(f"[bold]Encrypted: {encrypted}[/bold]")
        return encrypted

    def decrypt(self, ciphertext: List[int]) -> str:
        """Decrypt the ciphertext"""
        if len(ciphertext) != 16:
            raise ValueError("Ciphertext must be a list of 16 integers.")
        state = [ciphertext[i:i + 4] for i in range(0, 16, 4)]
        key_matrix = [[self.key[i * 4 + j] for j in range(4)] for i in range(4)]

        print("[bold]Initial State (Ciphertext)[/bold]")
        self.print_state(state, "Initial State")
        self.print_state(key_matrix, "Key Matrix")

        state = self.add_round_key(state, key_matrix)
        state = self.shift_rows(state, inverse=True)
        state = self.sub_bytes(state, inv_sbox)
        state = self.add_round_key(state, key_matrix)

        decrypted = ''.join(chr(byte) for row in state for byte in row)
        print(f"[bold]Decrypted: {decrypted}[/bold]")
        return decrypted

    def print_state(self, state: List[List[int]], step: str):
        """Print the state matrix in a readable format"""
        table = Table(title=step)
        table.add_column("Column 1")
        table.add_column("Column 2")
        table.add_column("Column 3")
        table.add_column("Column 4")
        for row in state:
            table.add_row(*[f"{x:02x}" for x in row])
        print(table)

# Example usage
if __name__ == "__main__":
    key = "a" * 16
    aes16 = AES16(key)
    plaintext = "simpleplaintext!"
    encrypted = aes16.encrypt(plaintext)
    decrypted = aes16.decrypt(encrypted)

