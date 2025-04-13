from .enigma import Plugboard, Rotor, Reflector, Enigma


def test_plugboard_swap():
    plugboard = Plugboard([("A", "B"), ("C", "D")])
    assert plugboard.swap("A") == "B"
    assert plugboard.swap("B") == "A"
    assert plugboard.swap("C") == "D"
    assert plugboard.swap("D") == "C"
    assert plugboard.swap("E") == "E"  # No swap for 'E'


def test_rotor_encipher_forward():
    rotor = Rotor("EKMFLGDQVZNTOWYHXUSPAIBRCJ", "Q")
    assert rotor.encipher_forward("A") == "E"
    assert rotor.encipher_forward("B") == "K"
    assert rotor.encipher_forward("C") == "M"


def test_rotor_encipher_backward():
    rotor = Rotor("EKMFLGDQVZNTOWYHXUSPAIBRCJ", "Q")
    assert rotor.encipher_backward("E") == "A"
    assert rotor.encipher_backward("K") == "B"
    assert rotor.encipher_backward("M") == "C"


def test_reflector_reflect():
    reflector = Reflector("YRUHQSLDPXNGOKMIEBFZCWVJAT")
    assert reflector.reflect("A") == "Y"
    assert reflector.reflect("B") == "R"
    assert reflector.reflect("C") == "U"


def test_enigma_encipher_single_character():
    plugboard = Plugboard([("A", "Z"), ("S", "O")])
    rotor1 = Rotor("EKMFLGDQVZNTOWYHXUSPAIBRCJ", "Q")
    rotor2 = Rotor("AJDKSIRUXBLHWTMCQGZNPYFVOE", "E")
    rotor3 = Rotor("BDFHJLCPRTXVZNYEIWGAKMUSQO", "V")
    rotors = [rotor1, rotor2, rotor3]
    reflector = Reflector("YRUHQSLDPXNGOKMIEBFZCWVJAT")
    enigma = Enigma(plugboard, rotors, reflector)

    assert enigma.encipher("S") == "K"
