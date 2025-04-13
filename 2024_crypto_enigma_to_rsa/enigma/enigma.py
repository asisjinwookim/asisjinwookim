from typing import List, Tuple, Dict


class Plugboard:
    def __init__(self, connections: List[Tuple[str, str]]):
        self.connections: Dict[str, str] = {}
        for a, b in connections:
            self.connections[a] = b
            self.connections[b] = a

    def swap(self, char: str) -> str:
        return self.connections.get(char, char)


class Rotor:
    def __init__(self, wiring: str, notch: str, position: int = 0):
        self.wiring = wiring
        self.notch = notch
        self.position = position

    def encipher_forward(self, char: str) -> str:
        index = (ord(char) - ord("A") + self.position) % 26
        return chr((ord(self.wiring[index]) - ord("A") - self.position) % 26 + ord("A"))

    def encipher_backward(self, char: str) -> str:
        index = (
            self.wiring.index(
                chr((ord(char) - ord("A") + self.position) % 26 + ord("A"))
            )
            - self.position
        ) % 26
        return chr(index + ord("A"))

    def rotate(self) -> bool:
        self.position = (self.position + 1) % 26
        return self.wiring[self.position] == self.notch


class Reflector:
    def __init__(self, wiring: str):
        self.wiring = wiring

    def reflect(self, char: str) -> str:
        index = ord(char) - ord("A")
        return self.wiring[index]


class Enigma:
    def __init__(self, plugboard: Plugboard, rotors: List[Rotor], reflector: Reflector):
        self.plugboard = plugboard
        self.rotors = rotors
        self.reflector = reflector

    def encipher(self, text: str) -> str:
        result = []
        for index, char in enumerate(text):
            if not char.isalpha():
                result.append(char)
                continue

            print(f"Encipher index: #{index}")
            char = char.upper()
            print(f"Initial character: {char}")

            char = self.plugboard.swap(char)
            print(f"After plugboard (forward): {char}")

            for rotor in self.rotors:
                char = rotor.encipher_forward(char)
                print(f"After rotor (forward) {rotor.position}: {char}")

            char = self.reflector.reflect(char)
            print(f"After reflector: {char}")

            for rotor in reversed(self.rotors):
                char = rotor.encipher_backward(char)
                print(f"After rotor (backward) {rotor.position}: {char}")

            char = self.plugboard.swap(char)
            print(f"After plugboard (backward): {char}\n")

            result.append(char)

            for i in range(len(self.rotors)):
                if not self.rotors[i].rotate():
                    break

        return "".join(result)


def main():
    # Fixed settings
    plugboard_connections = [("A", "Z"), ("S", "O")]
    plugboard = Plugboard(plugboard_connections)

    rotor1 = Rotor("EKMFLGDQVZNTOWYHXUSPAIBRCJ", "Q")
    rotor2 = Rotor("AJDKSIRUXBLHWTMCQGZNPYFVOE", "E")
    rotor3 = Rotor("BDFHJLCPRTXVZNYEIWGAKMUSQO", "V")
    rotors = [rotor1, rotor2, rotor3]

    reflector_wiring = "YRUHQSLDPXNGOKMIEBFZCWVJAT"
    reflector = Reflector(reflector_wiring)

    # Get user input for the text to encipher
    text = input("Enter text to encipher: ")

    enigma = Enigma(plugboard, rotors, reflector)
    ciphertext = enigma.encipher(text)
    print(f"Ciphertext: {ciphertext}")


if __name__ == "__main__":
    main()
