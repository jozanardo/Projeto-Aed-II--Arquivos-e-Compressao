import heapq
import os
import cv2
from collections import OrderedDict
from collections import Counter

"""
author: Bhrigu Srivastava
website: https:bhrigu.me
"""


class HuffmanCoding:
    def __init__(self, path):
        self.path = path
        self.heap = []
        self.codes = {}
        self.reverse_mapping = {}

    class HeapNode:
        def __init__(self, char, freq):
            self.char = char
            self.freq = freq
            self.left = None
            self.right = None

        # defining comparators less_than and equals
        def __lt__(self, other):
            return self.freq < other.freq

        def __eq__(self, other):
            if (other == None):
                return False
            if (not isinstance(other, HeapNode)):
                return False
            return self.freq == other.freq

    # functions for compression:

    def make_frequency_dict(self, text):
        frequency = {}
        for character in text:
            if not character in frequency:
                frequency[character] = 0
            frequency[character] += 1
        return frequency

    def make_heap(self, frequency):
        for key in frequency:
            node = self.HeapNode(key, frequency[key])
            heapq.heappush(self.heap, node)

    def merge_nodes(self):
        while (len(self.heap) > 1):
            node1 = heapq.heappop(self.heap)
            node2 = heapq.heappop(self.heap)

            merged = self.HeapNode(None, node1.freq + node2.freq)
            merged.left = node1
            merged.right = node2

            heapq.heappush(self.heap, merged)

    def make_codes_helper(self, root, current_code):
        if (root == None):
            return

        if (root.char != None):
            self.codes[root.char] = current_code
            self.reverse_mapping[current_code] = root.char
            return

        self.make_codes_helper(root.left, current_code + "0")
        self.make_codes_helper(root.right, current_code + "1")

    def make_codes(self):
        root = heapq.heappop(self.heap)
        current_code = ""
        self.make_codes_helper(root, current_code)

    def get_encoded_text(self, teste):
        encoded_text = ""
        for character in teste:
            encoded_text += self.codes[str(character)]
        return encoded_text

    def pad_encoded_bits(self, encoded_bits):
        extra_padding = 8 - len(encoded_bits) % 8
        for i in range(extra_padding):
            encoded_bits += "0"

        padded_info = "{0:08b}".format(extra_padding)
        encoded_bits = padded_info + encoded_bits
        return encoded_bits

    def get_byte_array(self, padded_encoded_text):
        if (len(padded_encoded_text) % 8 != 0):
            print("Encoded text not padded properly")
            exit(0)

        b = bytearray()
        for i in range(0, len(padded_encoded_text), 8):
            byte = padded_encoded_text[i:i + 8]
            b.append(int(byte, 2))
        return b

    def CountFrequency(self, teste):
        freq = {}

        for item in teste:
            if str(item) in freq:
                freq[str(item)] += 1

            else:
                freq[str(item)] = 1
        return freq

    def compress(self):
        filename, file_extension = os.path.splitext(self.path)
        output_path = filename + "2.bin"
        cap = cv2.VideoCapture('test.mp4')
        #cap = cv2.VideoCapture('https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/720/Big_Buck_Bunny_720_10s_10MB.mp4')
        gray = []
        frames = []

        with open(self.path, 'r+') as file, open(output_path, 'wb') as output:
            while (cap.isOpened()):
                ret, frame = cap.read()

                if ret == True:
                    frames.append(frame)
                    #converte os frames em tons e cinza
                    gray.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                    #break esta aqui só para pegar primeiro frame, retirar
                    break
                else:
                    break

            for i in  range(len(gray)):
                dicionario = dict(sum(map(Counter, gray[i]), Counter()))


            #ordena o dicionario de acordo da menor frequencia para maior
            #dicionario = OrderedDict(sorted(dicionario.items(), key=lambda x: x[1]))

            self.make_heap(dicionario)
            self.merge_nodes()
            self.make_codes()

            bits=""

            #percorre todos valores de cinza e converter os mesmos para bits
            for x in gray:
                for y in x:
                    for z in y:
                        bits += self.codes[z]


            padded_encoded_bits = self.pad_encoded_bits(bits)

            b = self.get_byte_array(padded_encoded_bits)
            output.write(bytes(b))

        print("Compressed")
        return output_path

    """ functions for decompression: """

    def remove_padding(self, padded_encoded_text):
        padded_info = padded_encoded_text[:8]
        extra_padding = int(padded_info, 2)

        padded_encoded_text = padded_encoded_text[8:]
        encoded_text = padded_encoded_text[:-1 * extra_padding]

        return encoded_text

    def decode_text(self, encoded_text):
        current_code = ""
        decoded_text = ""

        for bit in encoded_text:
            current_code += bit
            if (current_code in self.reverse_mapping):
                character = self.reverse_mapping[current_code]
                decoded_text += character
                current_code = ""

        return decoded_text

    def decompress(self, input_path):
        filename, file_extension = os.path.splitext(self.path)
        output_path = filename + "_decompressed" + ".png"

        with open(input_path, 'rb') as file, open(output_path, 'w') as output:
            bit_string = ""

            byte = file.read(1)
            while (len(byte) > 0):
                byte = ord(byte)
                bits = bin(byte)[2:].rjust(8, '0')
                bit_string += bits
                byte = file.read(1)

            encoded_text = self.remove_padding(bit_string)

            decompressed_text = self.decode_text(encoded_text)

            output.write(decompressed_text)

        print("Decompressed")
        return output_path