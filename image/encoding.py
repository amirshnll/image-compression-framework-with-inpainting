import base64
from heapq import heappush, heappop, heapify
from collections import defaultdict
import ast


class Encoding:
    def encode(self, data: str | bytes, method: str) -> str:
        """
        Encodes the input data using the specified encoding method.
        """
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        if method == "rle":
            return self.rle_encode(data)
        elif method == "hamming":
            return self.hamming_encode(data)
        elif method == "huffman":
            return self.huffman_encode(data)
        elif method == "base64":
            return self.base64_encode(data)
        elif method == "ascii":
            return self.ascii_encode(data)
        else:
            raise ValueError(f"Unsupported encoding method: {method}")

    def decode(self, encoded_data: str, method: str) -> str:
        """
        Decodes the input data using the specified decoding method.
        """
        if isinstance(encoded_data, bytes):
            encoded_data = encoded_data.decode("utf-8")
        if method == "rle":
            return self.rle_decode(encoded_data)
        elif method == "hamming":
            return self.hamming_decode(encoded_data)
        elif method == "huffman":
            return self.huffman_decode(encoded_data)
        elif method == "base64":
            return self.base64_decode(encoded_data)
        elif method == "ascii":
            return self.ascii_decode(encoded_data)
        else:
            raise NotImplementedError(f"Decoding method not implemented for: {method}")

    # 1. Run-Length Encoding (RLE)
    # Reference: G. M. Adelson-Velskii, V. L. Arlazarov, and M. A. Kronrod, "Pattern recognition," 1967.
    def rle_encode(self, data: str) -> list:
        """
        Encodes the input data using the Run-Length Encoding (RLE) algorithm.
        """
        encoding = []
        prev_char = ""
        count = 1

        if not data:
            return ""

        for char in data:
            if char != prev_char:
                if prev_char:
                    encoding.append((prev_char, count))
                count = 1
                prev_char = char
            else:
                count += 1
        encoding.append((prev_char, count))
        return encoding

    def rle_decode(self, encoded_data: list) -> str:
        """
        Decodes the input RLE-encoded data.
        """
        if isinstance(encoded_data, str):
            encoded_data = ast.literal_eval(encoded_data)

        decoded_data = ""
        for char, count in encoded_data:
            decoded_data += char * count
        return decoded_data

    # 2. Hamming Encoding
    # Reference: R. W. Hamming, "Error Detecting and Error Correcting Codes," Bell System Technical Journal, 1950.
    def hamming_encode(self, data: str) -> str:
        """
        Encodes the input data using the Hamming code.
        """

        def add_parity_bits(bits: list) -> list:
            p1 = bits[0] ^ bits[1] ^ bits[3]
            p2 = bits[0] ^ bits[2] ^ bits[3]
            p3 = bits[1] ^ bits[2] ^ bits[3]
            return [p1, p2, bits[0], p3, bits[1], bits[2], bits[3]]

        binary_data = "".join(format(ord(char), "08b") for char in data)
        encoded_data = ""

        for i in range(0, len(binary_data), 4):
            chunk = [int(bit) for bit in binary_data[i : i + 4]]
            while len(chunk) < 4:
                chunk.append(0)
            encoded_chunk = add_parity_bits(chunk)
            encoded_data += "".join(map(str, encoded_chunk))

        return encoded_data

    def hamming_decode(self, encoded_data: str) -> str:
        """
        Decodes the input Hamming-encoded data.
        """

        def decode_chunk(bits: list) -> list:
            p1 = bits[0] ^ bits[2] ^ bits[4] ^ bits[6]
            p2 = bits[1] ^ bits[2] ^ bits[5] ^ bits[6]
            p3 = bits[3] ^ bits[4] ^ bits[5] ^ bits[6]
            error_pos = p1 * 1 + p2 * 2 + p3 * 4
            if error_pos != 0:
                bits[error_pos - 1] ^= 1
            return [bits[2], bits[4], bits[5], bits[6]]

        decoded_bits = []
        for i in range(0, len(encoded_data), 7):
            chunk = [int(bit) for bit in encoded_data[i : i + 7]]
            decoded_bits.extend(decode_chunk(chunk))

        decoded_chars = [
            chr(int("".join(map(str, decoded_bits[i : i + 8])), 2))
            for i in range(0, len(decoded_bits), 8)
        ]
        return "".join(decoded_chars)

    # 3. Huffman Encoding
    # Reference: D. A. Huffman, "A Method for the Construction of Minimum-Redundancy Codes," Proceedings of the IRE, 1952.
    def huffman_encode(self, data: str) -> str:
        """
        Encodes the input data using the Huffman coding algorithm.
        This method constructs a frequency dictionary for the characters in the input data,
        builds a Huffman tree based on the character frequencies, generates Huffman codes
        for each character by traversing the tree, and encodes the input data using these
        codes. The encoded data and the Huffman codes are returned as a single string.
        """
        if not data:
            return ""

        frequency = defaultdict(int)
        for char in data:
            frequency[char] += 1

        heap = [Node(char, freq) for char, freq in frequency.items()]
        heapify(heap)

        while len(heap) > 1:
            node1 = heappop(heap)
            node2 = heappop(heap)
            merged = Node(None, node1.freq + node2.freq)
            merged.left = node1
            merged.right = node2
            heappush(heap, merged)

        huffman_tree = heap[0]

        huffman_codes = {}
        self.generate_huffman_codes(huffman_tree, "", huffman_codes)

        encoded_data = "".join(huffman_codes[char] for char in data)
        return encoded_data + "\n" + str(huffman_codes)

    def generate_huffman_codes(self, node, current_code, huffman_codes):
        """
        Generates Huffman codes for each character by traversing the Huffman tree.
        This is a recursive helper function that traverses the Huffman tree and assigns
        binary codes to each character. The codes are stored in the huffman_codes dictionary.
        """
        if node is None:
            return

        if node.char is not None:
            huffman_codes[node.char] = current_code
            return

        self.generate_huffman_codes(node.left, current_code + "0", huffman_codes)
        self.generate_huffman_codes(node.right, current_code + "1", huffman_codes)

    def huffman_decode(self, encoded_data: str) -> str:
        """
        Decodes the Huffman encoded data using the Huffman codes.
        This method splits the encoded data into the Huffman codes and the encoded message.
        It reconstructs the Huffman codes dictionary, then decodes the message by traversing
        the encoded data and matching it with the Huffman codes.
        """
        encoded_data, huffman_codes_str = encoded_data.split("\n")
        huffman_codes = ast.literal_eval(huffman_codes_str)
        reverse_huffman_codes = {v: k for k, v in huffman_codes.items()}

        current_code = ""
        decoded_data = ""

        for bit in encoded_data:
            current_code += bit
            if current_code in reverse_huffman_codes:
                decoded_data += reverse_huffman_codes[current_code]
                current_code = ""

        return decoded_data

    # 4. Base64 Encoding
    # Reference: RFC 4648, "The Base16, Base32, and Base64 Data Encodings," 2006.
    def base64_encode(self, data: str) -> str:
        """
        Encodes the input data using the Base64 encoding scheme.
        """
        encoded_bytes = base64.b64encode(data.encode("utf-8"))
        encoded_str = str(encoded_bytes, "utf-8")
        return encoded_str

    def base64_decode(self, encoded_data: str) -> str:
        """
        Decodes the input Base64-encoded data.
        """
        decoded_data = base64.b64decode(encoded_data)
        decoded_data = decoded_data.decode("utf-8")
        return decoded_data

    # 5. ASCII Encoding
    # Reference: American Standard Code for Information Interchange (ASCII), ANSI X3.4, 1963.
    def ascii_encode(self, data: str) -> str:
        """
        Encodes the input data using the ASCII (American Standard Code for Information Interchange) encoding scheme.
        """
        encoded = [ord(char) for char in data]
        return str(encoded)

    def ascii_decode(self, encoded_data: str) -> str:
        """ "
        Decodes the input ASCII-encoded data.
        """
        encoded_data = ast.literal_eval(encoded_data)
        decoded_data = "".join([chr(byte) for byte in encoded_data])
        return decoded_data


class Node:
    def __init__(self, char: str | None, freq: list) -> None:
        """
        Initializes the class instance.
        """
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other: list) -> bool:
        """
        Compares two Node objects based on their frequencies.
        This method is used to sort the nodes in ascending order of frequency.
        It returns True if this node has a lower frequency than the other node,
        and False otherwise.
        """
        return self.freq < other.freq
